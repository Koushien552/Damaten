#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace hexai {

constexpr int EMPTY = 0;
constexpr int BLACK = 1;
constexpr int WHITE = 2;
constexpr double WIN_SCORE = 1000000.0;

int other_player(int p) { return p == BLACK ? WHITE : BLACK; }

bool is_integer_text(const std::string& s) {
    if (s.empty()) return false;
    std::size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    if (i == s.size()) return false;
    for (; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(s);
    while (std::getline(ss, item, delim)) out.push_back(item);
    return out;
}

struct Random {
    std::mt19937_64 engine;

    explicit Random(uint64_t seed = 0) {
        if (seed == 0) {
            seed = static_cast<uint64_t>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count());
        }
        engine.seed(seed);
    }

    int uniform_int(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(engine);
    }

    double uniform01() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(engine);
    }

    double normal(double mean, double stdev) {
        std::normal_distribution<double> dist(mean, stdev);
        return dist(engine);
    }

    double gamma(double alpha) {
        std::gamma_distribution<double> dist(alpha, 1.0);
        return dist(engine);
    }
};

Random g_rng(123456789ULL);

struct Rules {
    int n = 9;
    int nn = 81;
    int b_top = 81;
    int b_bottom = 82;
    int w_left = 83;
    int w_right = 84;
    std::vector<std::array<int, 6>> nbrs;
    std::vector<int> nbr_count;

    explicit Rules(int board_size = 9) : n(board_size), nn(board_size * board_size) {
        b_top = nn;
        b_bottom = nn + 1;
        w_left = nn + 2;
        w_right = nn + 3;
        nbrs.assign(nn, {});
        nbr_count.assign(nn, 0);
        build_neighbors();
    }

    void build_neighbors() {
        const int dr[6] = {-1, -1, 0, 0, 1, 1};
        const int dc[6] = {0, 1, -1, 1, -1, 0};
        for (int i = 0; i < nn; ++i) {
            int r = i / n;
            int c = i % n;
            int cnt = 0;
            for (int d = 0; d < 6; ++d) {
                int nr = r + dr[d];
                int nc = c + dc[d];
                if (nr >= 0 && nr < n && nc >= 0 && nc < n) {
                    nbrs[i][cnt++] = nr * n + nc;
                }
            }
            nbr_count[i] = cnt;
        }
    }

    std::string move_to_string(int m) const {
        if (m < 0 || m >= nn) return "pass";
        return std::string(1, static_cast<char>('a' + (m % n))) + std::to_string(m / n + 1);
    }

    int move_from_string(const std::string& raw) const {
        if (raw.size() < 2) return -1;
        int c = std::tolower(static_cast<unsigned char>(raw[0])) - 'a';
        int r = -1;
        try {
            r = std::stoi(raw.substr(1)) - 1;
        } catch (...) {
            return -1;
        }
        if (r < 0 || r >= n || c < 0 || c >= n) return -1;
        return r * n + c;
    }
};

struct UnionFind {
    std::vector<int> parent;
    std::vector<int> rank;

    explicit UnionFind(int size = 0) { reset(size); }

    void reset(int size) {
        parent.resize(size);
        rank.assign(size, 0);
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank[a] < rank[b]) std::swap(a, b);
        parent[b] = a;
        if (rank[a] == rank[b]) ++rank[a];
    }

    bool connected(int a, int b) { return find(a) == find(b); }

    // Non-mutating variants (no path compression) so callers that only need to
    // test connectivity can stay const and avoid copying the whole structure.
    int find_const(int x) const {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    bool connected_const(int a, int b) const { return find_const(a) == find_const(b); }
};

void uf_place(const Rules& rules, UnionFind& uf, const std::vector<int>& board, int idx, int color) {
    int r = idx / rules.n;
    int c = idx % rules.n;
    if (color == BLACK) {
        if (r == 0) uf.unite(idx, rules.b_top);
        if (r == rules.n - 1) uf.unite(idx, rules.b_bottom);
    } else {
        if (c == 0) uf.unite(idx, rules.w_left);
        if (c == rules.n - 1) uf.unite(idx, rules.w_right);
    }
    for (int k = 0; k < rules.nbr_count[idx]; ++k) {
        int nb = rules.nbrs[idx][k];
        if (board[nb] == color) uf.unite(idx, nb);
    }
}

struct GameState {
    std::shared_ptr<const Rules> rules;
    std::vector<int> board;
    int cur = BLACK;
    int last_move = -1;
    UnionFind uf;

    explicit GameState(std::shared_ptr<const Rules> r = std::make_shared<Rules>(9))
        : rules(std::move(r)), board(rules->nn, EMPTY), cur(BLACK), last_move(-1), uf(rules->nn + 4) {}

    GameState apply(int move) const {
        GameState ns = *this;
        ns.board[move] = cur;
        uf_place(*ns.rules, ns.uf, ns.board, move, cur);
        ns.cur = other_player(cur);
        ns.last_move = move;
        return ns;
    }

    bool has_won(int player) const {
        if (player == BLACK) return uf.connected_const(rules->b_top, rules->b_bottom);
        return uf.connected_const(rules->w_left, rules->w_right);
    }

    bool is_terminal() const { return has_won(BLACK) || has_won(WHITE); }

    int winner() const {
        if (has_won(BLACK)) return BLACK;
        if (has_won(WHITE)) return WHITE;
        return EMPTY;
    }

    int empty_count() const {
        int count = 0;
        for (int v : board) {
            if (v == EMPTY) ++count;
        }
        return count;
    }
};

std::vector<int> legal_moves(const GameState& gs) {
    std::vector<int> moves;
    moves.reserve(gs.rules->nn);
    for (int i = 0; i < gs.rules->nn; ++i) {
        if (gs.board[i] == EMPTY) moves.push_back(i);
    }
    return moves;
}

void rebuild_union_find(GameState& gs) {
    gs.uf.reset(gs.rules->nn + 4);
    for (int i = 0; i < gs.rules->nn; ++i) {
        if (gs.board[i] != EMPTY) uf_place(*gs.rules, gs.uf, gs.board, i, gs.board[i]);
    }
}

std::string board_to_string(const std::vector<int>& board) {
    std::string s;
    s.reserve(board.size());
    for (int v : board) {
        if (v == BLACK) s.push_back('B');
        else if (v == WHITE) s.push_back('W');
        else s.push_back('.');
    }
    return s;
}

bool parse_board_text(const std::string& text, int nn, std::vector<int>& out) {
    out.assign(nn, EMPTY);
    std::string compact;
    for (char ch : text) {
        if (!std::isspace(static_cast<unsigned char>(ch)) && ch != ',' && ch != '|') {
            compact.push_back(ch);
        }
    }
    if (static_cast<int>(compact.size()) != nn) return false;
    for (int i = 0; i < nn; ++i) {
        char ch = compact[i];
        if (ch == '.' || ch == '0' || ch == '-') out[i] = EMPTY;
        else if (ch == 'B' || ch == 'b' || ch == '1' || ch == 'X' || ch == 'x') out[i] = BLACK;
        else if (ch == 'W' || ch == 'w' || ch == '2' || ch == 'O' || ch == 'o') out[i] = WHITE;
        else return false;
    }
    return true;
}

void print_board(const GameState& gs) {
    const Rules& r = *gs.rules;
    std::cout << "\n  ";
    for (int c = 0; c < r.n; ++c) std::cout << static_cast<char>('a' + c) << ' ';
    std::cout << "\n";
    for (int row = 0; row < r.n; ++row) {
        std::cout << std::string(row, ' ') << std::setw(2) << (row + 1) << ' ';
        for (int col = 0; col < r.n; ++col) {
            int v = gs.board[row * r.n + col];
            std::cout << (v == BLACK ? 'B' : (v == WHITE ? 'W' : '.')) << ' ';
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

bool wins_if_played(const GameState& gs, int move, int player) {
    if (move < 0 || move >= gs.rules->nn || gs.board[move] != EMPTY) return false;
    std::vector<int> board = gs.board;
    UnionFind uf = gs.uf;
    board[move] = player;
    uf_place(*gs.rules, uf, board, move, player);
    if (player == BLACK) return uf.connected(gs.rules->b_top, gs.rules->b_bottom);
    return uf.connected(gs.rules->w_left, gs.rules->w_right);
}

void find_critical_cells(const GameState& gs, std::vector<int>& winning, std::vector<int>& must_block) {
    winning.clear();
    must_block.clear();
    int me = gs.cur;
    int opp = other_player(me);
    for (int m = 0; m < gs.rules->nn; ++m) {
        if (gs.board[m] != EMPTY) continue;
        if (wins_if_played(gs, m, me)) {
            winning.push_back(m);
        } else if (wins_if_played(gs, m, opp)) {
            must_block.push_back(m);
        }
    }
}

std::vector<int> get_bridge_saves(const Rules& rules,
                                  const std::vector<int>& board,
                                  int player,
                                  int last_opp) {
    if (last_opp < 0 || last_opp >= rules.nn) return {};

    // The opponent's intruding stone is treated as empty when measuring the
    // bridge it just stepped into.
    std::vector<int> before = board;
    before[last_opp] = EMPTY;

    // A bridge carrier threatened by last_opp must have both of its endpoints
    // adjacent to last_opp, so only the player's stones next to it can matter.
    std::array<int, 6> cand{};
    int cand_n = 0;
    for (int k = 0; k < rules.nbr_count[last_opp]; ++k) {
        int nb = rules.nbrs[last_opp][k];
        if (board[nb] == player) cand[cand_n++] = nb;
    }

    std::vector<int> mark(rules.nn, 0);
    for (int ia = 0; ia < cand_n; ++ia) {
        int a = cand[ia];
        for (int ib = ia + 1; ib < cand_n; ++ib) {
            int b = cand[ib];

            bool adjacent = false;
            for (int k = 0; k < rules.nbr_count[a]; ++k) {
                if (rules.nbrs[a][k] == b) {
                    adjacent = true;
                    break;
                }
            }
            if (adjacent) continue;

            std::array<int, 6> commons{};
            int cn = 0;
            bool has_last = false;
            for (int ka = 0; ka < rules.nbr_count[a]; ++ka) {
                int na = rules.nbrs[a][ka];
                if (before[na] != EMPTY) continue;
                bool shared = false;
                for (int kb = 0; kb < rules.nbr_count[b]; ++kb) {
                    if (rules.nbrs[b][kb] == na) {
                        shared = true;
                        break;
                    }
                }
                if (!shared) continue;
                if (na == last_opp) has_last = true;
                if (cn < 6) commons[cn++] = na;
            }
            if (cn < 2 || !has_last) continue;
            for (int t = 0; t < cn; ++t) {
                int x = commons[t];
                if (x != last_opp && board[x] == EMPTY) mark[x] = 1;
            }
        }
    }

    std::vector<int> result;
    for (int i = 0; i < rules.nn; ++i) {
        if (mark[i]) result.push_back(i);
    }
    return result;
}

int connection_cost(const Rules& rules, const std::vector<int>& board, int player) {
    constexpr int INF = 100000000;
    auto cell_cost = [&](int idx) {
        if (board[idx] == player) return 0;
        if (board[idx] == EMPTY) return 1;
        return INF;
    };

    std::vector<int> dist(rules.nn, INF);
    using Item = std::pair<int, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;

    if (player == BLACK) {
        for (int c = 0; c < rules.n; ++c) {
            int idx = c;
            int cost = cell_cost(idx);
            if (cost < dist[idx]) {
                dist[idx] = cost;
                pq.push({cost, idx});
            }
        }
    } else {
        for (int row = 0; row < rules.n; ++row) {
            int idx = row * rules.n;
            int cost = cell_cost(idx);
            if (cost < dist[idx]) {
                dist[idx] = cost;
                pq.push({cost, idx});
            }
        }
    }

    while (!pq.empty()) {
        auto [d, v] = pq.top();
        pq.pop();
        if (d != dist[v]) continue;
        for (int k = 0; k < rules.nbr_count[v]; ++k) {
            int nb = rules.nbrs[v][k];
            int nd = d + cell_cost(nb);
            if (nd < dist[nb]) {
                dist[nb] = nd;
                pq.push({nd, nb});
            }
        }
    }

    int best = INF;
    if (player == BLACK) {
        for (int c = 0; c < rules.n; ++c) best = std::min(best, dist[(rules.n - 1) * rules.n + c]);
    } else {
        for (int row = 0; row < rules.n; ++row) best = std::min(best, dist[row * rules.n + rules.n - 1]);
    }
    return best;
}

double heuristic_black_win(const GameState& gs) {
    if (gs.has_won(BLACK)) return 1.0;
    if (gs.has_won(WHITE)) return 0.0;
    int cb = connection_cost(*gs.rules, gs.board, BLACK);
    int cw = connection_cost(*gs.rules, gs.board, WHITE);
    double x = std::max(-10.0, std::min(10.0, (cw - cb) * 0.85));
    return 1.0 / (1.0 + std::exp(-x));
}

double rollout_light(const GameState& start) {
    GameState st = start;
    std::vector<int> empty = legal_moves(st);
    std::shuffle(empty.begin(), empty.end(), g_rng.engine);

    auto take_move = [&](int move) {
        auto it = std::find(empty.begin(), empty.end(), move);
        if (it != empty.end()) {
            *it = empty.back();
            empty.pop_back();
        }
    };

    while (!empty.empty()) {
        int move = -1;
        auto saves = get_bridge_saves(*st.rules, st.board, st.cur, st.last_move);
        for (int s : saves) {
            if (st.board[s] == EMPTY) {
                move = s;
                break;
            }
        }
        if (move >= 0) {
            take_move(move);
        } else {
            move = empty.back();
            empty.pop_back();
        }

        st = st.apply(move);
        if (st.has_won(BLACK)) return 1.0;
        if (st.has_won(WHITE)) return 0.0;
    }

    return st.has_won(BLACK) ? 1.0 : 0.0;
}

double evaluate_for_current_player(const GameState& gs) {
    double bp = heuristic_black_win(gs);
    double current_prob = gs.cur == BLACK ? bp : (1.0 - bp);
    return current_prob * 2.0 - 1.0;
}

std::vector<int> order_moves_for_ab(const GameState& gs) {
    std::vector<int> winning;
    std::vector<int> blocks;
    find_critical_cells(gs, winning, blocks);
    if (!winning.empty()) return {winning.front()};

    std::set<int> block_set(blocks.begin(), blocks.end());
    auto saves = get_bridge_saves(*gs.rules, gs.board, gs.cur, gs.last_move);
    std::set<int> save_set(saves.begin(), saves.end());
    std::vector<int> moves = legal_moves(gs);

    const double center = (gs.rules->n - 1) / 2.0;
    std::sort(moves.begin(), moves.end(), [&](int a, int b) {
        auto score = [&](int m) {
            double s = 0.0;
            if (block_set.count(m)) s += 100.0;
            if (save_set.count(m)) s += 15.0;
            s -= std::abs(m / gs.rules->n - center) + std::abs(m % gs.rules->n - center);
            return s;
        };
        return score(a) > score(b);
    });
    return moves;
}

double alphabeta(GameState gs, int depth, double alpha, double beta) {
    if (gs.has_won(other_player(gs.cur))) return -WIN_SCORE - depth;
    if (depth <= 0) return evaluate_for_current_player(gs);

    auto moves = order_moves_for_ab(gs);
    if (moves.empty()) return 0.0;

    double best = -std::numeric_limits<double>::infinity();
    for (int m : moves) {
        GameState ns = gs.apply(m);
        double value = -alphabeta(ns, depth - 1, -beta, -alpha);
        best = std::max(best, value);
        alpha = std::max(alpha, best);
        if (alpha >= beta) break;
    }
    return best;
}

int alphabeta_best_move(const GameState& gs, int depth) {
    auto moves = order_moves_for_ab(gs);
    if (moves.empty()) return -1;
    if (moves.size() == 1) return moves.front();

    int best_move = moves.front();
    double best_value = -std::numeric_limits<double>::infinity();
    for (int m : moves) {
        GameState ns = gs.apply(m);
        double value = -alphabeta(ns, depth - 1, -WIN_SCORE, WIN_SCORE);
        if (value > best_value) {
            best_value = value;
            best_move = m;
        }
    }
    return best_move;
}

struct NetForward {
    std::vector<double> x;
    std::vector<double> z1;
    std::vector<double> a1;
    std::vector<double> z2;
    std::vector<double> a2;
    std::vector<double> logits;
    std::vector<double> policy;
    double value = 0.0;
};

// Result of a single network evaluation, shared by every model type.
struct Eval {
    std::vector<double> logits;  // raw policy logits over all nn cells
    double value = 0.0;          // tanh value from the current player's view
};

// Common interface so the search can use either the small MLP (TinyNet) or the
// convolutional residual network (ConvNet) interchangeably.
struct NeuralNet {
    virtual ~NeuralNet() = default;
    virtual bool is_ready() const = 0;
    virtual int board_n() const = 0;
    virtual Eval evaluate(const std::vector<int>& board, int player) const = 0;
};

struct TinyNet : NeuralNet {
    int n = 0;
    int nn = 0;
    int input = 0;
    int h1 = 128;
    int h2 = 64;
    bool ready = false;
    std::vector<double> w1, b1, w2, b2, wp, bp, wv;
    double bv = 0.0;

    void init(int board_size, int hidden1 = 128, int hidden2 = 64, uint64_t seed = 20240531ULL) {
        n = board_size;
        nn = n * n;
        input = 2 * nn + 1;
        h1 = hidden1;
        h2 = hidden2;
        Random rr(seed);
        auto randn = [&](double scale) { return rr.normal(0.0, scale); };
        w1.assign(h1 * input, 0.0);
        b1.assign(h1, 0.0);
        w2.assign(h2 * h1, 0.0);
        b2.assign(h2, 0.0);
        wp.assign(nn * h2, 0.0);
        bp.assign(nn, 0.0);
        wv.assign(h2, 0.0);
        bv = 0.0;

        double s1 = std::sqrt(2.0 / std::max(1, input));
        double s2 = std::sqrt(2.0 / std::max(1, h1));
        double s3 = std::sqrt(2.0 / std::max(1, h2));
        for (double& v : w1) v = randn(s1);
        for (double& v : w2) v = randn(s2);
        for (double& v : wp) v = randn(s3);
        for (double& v : wv) v = randn(s3);
        ready = true;
    }

    std::vector<double> make_features(const std::vector<int>& board, int player) const {
        std::vector<double> x(input, 0.0);
        int opp = other_player(player);
        for (int i = 0; i < nn; ++i) {
            if (board[i] == player) x[i] = 1.0;
            else if (board[i] == opp) x[nn + i] = 1.0;
        }
        x[2 * nn] = player == BLACK ? 1.0 : -1.0;
        return x;
    }

    NetForward forward_raw(const std::vector<double>& x) const {
        NetForward f;
        f.x = x;
        f.z1.assign(h1, 0.0);
        f.a1.assign(h1, 0.0);
        f.z2.assign(h2, 0.0);
        f.a2.assign(h2, 0.0);
        f.logits.assign(nn, 0.0);
        f.policy.assign(nn, 0.0);

        for (int i = 0; i < h1; ++i) {
            double z = b1[i];
            const int base = i * input;
            for (int j = 0; j < input; ++j) z += w1[base + j] * x[j];
            f.z1[i] = z;
            f.a1[i] = std::tanh(z);
        }
        for (int i = 0; i < h2; ++i) {
            double z = b2[i];
            const int base = i * h1;
            for (int j = 0; j < h1; ++j) z += w2[base + j] * f.a1[j];
            f.z2[i] = z;
            f.a2[i] = std::tanh(z);
        }
        for (int o = 0; o < nn; ++o) {
            double z = bp[o];
            const int base = o * h2;
            for (int j = 0; j < h2; ++j) z += wp[base + j] * f.a2[j];
            f.logits[o] = z;
        }
        double vz = bv;
        for (int j = 0; j < h2; ++j) vz += wv[j] * f.a2[j];
        f.value = std::tanh(vz);

        double max_logit = *std::max_element(f.logits.begin(), f.logits.end());
        double sum = 0.0;
        for (int i = 0; i < nn; ++i) {
            f.policy[i] = std::exp(f.logits[i] - max_logit);
            sum += f.policy[i];
        }
        if (sum > 0.0) {
            for (double& p : f.policy) p /= sum;
        }
        return f;
    }

    NetForward infer(const std::vector<int>& board, int player) const {
        return forward_raw(make_features(board, player));
    }

    bool is_ready() const override { return ready; }
    int board_n() const override { return n; }
    Eval evaluate(const std::vector<int>& board, int player) const override {
        NetForward f = infer(board, player);
        Eval e;
        e.logits = std::move(f.logits);
        e.value = f.value;
        return e;
    }

    bool save(const std::string& path) const {
        std::ofstream out(path);
        if (!out) return false;
        out << "HEXNN_V1 " << n << ' ' << h1 << ' ' << h2 << "\n";
        auto write_vec = [&](const std::vector<double>& v) {
            out << v.size();
            out << std::setprecision(17);
            for (double x : v) out << ' ' << x;
            out << "\n";
        };
        write_vec(w1);
        write_vec(b1);
        write_vec(w2);
        write_vec(b2);
        write_vec(wp);
        write_vec(bp);
        write_vec(wv);
        out << std::setprecision(17) << bv << "\n";
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream in(path);
        if (!in) return false;
        std::string tag;
        in >> tag >> n >> h1 >> h2;
        if (tag != "HEXNN_V1") return false;
        nn = n * n;
        input = 2 * nn + 1;
        auto read_vec = [&](std::vector<double>& v) {
            std::size_t size = 0;
            in >> size;
            v.assign(size, 0.0);
            for (double& x : v) in >> x;
        };
        read_vec(w1);
        read_vec(b1);
        read_vec(w2);
        read_vec(b2);
        read_vec(wp);
        read_vec(bp);
        read_vec(wv);
        in >> bv;
        ready = static_cast<int>(w1.size()) == h1 * input &&
                static_cast<int>(w2.size()) == h2 * h1 &&
                static_cast<int>(wp.size()) == nn * h2 &&
                static_cast<int>(wv.size()) == h2;
        return ready;
    }

    double train_one(const std::vector<int>& board,
                     int player,
                     const std::vector<double>& target_policy,
                     double target_value,
                     double lr,
                     double value_weight) {
        NetForward f = infer(board, player);

        std::vector<double> dlogits(nn, 0.0);
        double policy_loss = 0.0;
        for (int i = 0; i < nn; ++i) {
            double t = i < static_cast<int>(target_policy.size()) ? target_policy[i] : 0.0;
            dlogits[i] = f.policy[i] - t;
            if (t > 0.0) policy_loss -= t * std::log(std::max(1e-12, f.policy[i]));
        }
        double value_err = f.value - target_value;
        double value_loss = value_err * value_err;
        double dvalue_z = 2.0 * value_weight * value_err * (1.0 - f.value * f.value);

        std::vector<double> da2(h2, 0.0);
        for (int o = 0; o < nn; ++o) {
            const int base = o * h2;
            for (int j = 0; j < h2; ++j) da2[j] += wp[base + j] * dlogits[o];
        }
        for (int j = 0; j < h2; ++j) da2[j] += wv[j] * dvalue_z;

        for (int o = 0; o < nn; ++o) {
            const int base = o * h2;
            for (int j = 0; j < h2; ++j) wp[base + j] -= lr * dlogits[o] * f.a2[j];
            bp[o] -= lr * dlogits[o];
        }
        for (int j = 0; j < h2; ++j) wv[j] -= lr * dvalue_z * f.a2[j];
        bv -= lr * dvalue_z;

        std::vector<double> dz2(h2, 0.0);
        for (int i = 0; i < h2; ++i) dz2[i] = da2[i] * (1.0 - f.a2[i] * f.a2[i]);

        std::vector<double> da1(h1, 0.0);
        for (int i = 0; i < h2; ++i) {
            const int base = i * h1;
            for (int j = 0; j < h1; ++j) da1[j] += w2[base + j] * dz2[i];
        }

        for (int i = 0; i < h2; ++i) {
            const int base = i * h1;
            for (int j = 0; j < h1; ++j) w2[base + j] -= lr * dz2[i] * f.a1[j];
            b2[i] -= lr * dz2[i];
        }

        std::vector<double> dz1(h1, 0.0);
        for (int i = 0; i < h1; ++i) dz1[i] = da1[i] * (1.0 - f.a1[i] * f.a1[i]);

        for (int i = 0; i < h1; ++i) {
            const int base = i * input;
            for (int j = 0; j < input; ++j) w1[base + j] -= lr * dz1[i] * f.x[j];
            b1[i] -= lr * dz1[i];
        }

        return policy_loss + value_weight * value_loss;
    }
};

// Convolutional residual network (AlphaZero-style) for Hex. Inference only on
// the C++ side; training happens in PyTorch (Colab) which exports the HEXCNN_V1
// format read here. No BatchNorm: residual blocks are plain conv+ReLU so the
// C++ forward pass stays simple and the exported weights need no folding.
//
// Layout (matches PyTorch default tensor ordering):
//   input planes [0]=my stones, [1]=opp stones, [2]=color (1 if Black to move)
//   trunk:  conv3x3(in_planes->C) + ReLU
//   B blocks: y = ReLU(x + conv3x3(ReLU(conv3x3(x))))
//   policy: conv1x1(C->2)+ReLU -> flatten(2*nn) -> FC(2*nn->nn)
//   value:  conv1x1(C->1)+ReLU -> flatten(nn) -> FC(nn->H)+ReLU -> FC(H->1)+tanh
struct ConvNet : NeuralNet {
    int n = 0, nn = 0;
    int C = 32, B = 4;
    int in_planes = 3, val_hidden = 64;
    bool ok = false;

    std::vector<float> w_in, b_in;                        // [C*in_planes*9], [C]
    std::vector<std::vector<float>> w1, bb1, w2, bb2;     // per block
    std::vector<float> w_ph, b_ph;                        // [2*C], [2]
    std::vector<float> w_pf, b_pf;                        // [nn*(2*nn)], [nn]
    std::vector<float> w_vh, b_vh;                        // [C], [1]
    std::vector<float> w_vf1, b_vf1;                      // [val_hidden*nn], [val_hidden]
    std::vector<float> w_vf2;                             // [val_hidden]
    float b_vf2 = 0.0f;

    bool is_ready() const override { return ok; }
    int board_n() const override { return n; }

    void make_planes(const std::vector<int>& board, int player, std::vector<float>& planes) const {
        planes.assign(in_planes * nn, 0.0f);
        int opp = other_player(player);
        for (int i = 0; i < nn; ++i) {
            if (board[i] == player) planes[i] = 1.0f;
            else if (board[i] == opp) planes[nn + i] = 1.0f;
        }
        float color = (player == BLACK) ? 1.0f : 0.0f;
        for (int i = 0; i < nn; ++i) planes[2 * nn + i] = color;
    }

    // 3x3 conv with zero padding 1. in:[in_ch*nn] out:[out_ch*nn]
    // w:[out_ch*in_ch*9] (oc,ic,kr,kc) row-major, b:[out_ch].
    // 3x3 conv, zero pad 1, implemented as im2col + GEMM. The weight layout
    // w[(oc*in_ch+ic)*9 + k] equals a [out_ch, in_ch*9] matrix, so the output
    // is W * cols where cols[(ic*9+k), cell] gathers the padded inputs. The
    // inner loop accumulates over contiguous cells (an element-wise AXPY, not a
    // reduction), so it vectorises cleanly under /arch:AVX2 with /fp:precise and
    // stays numerically equivalent to the scalar version up to FP rounding.
    void conv3x3(const std::vector<float>& in, int in_ch, int out_ch,
                 const std::vector<float>& w, const std::vector<float>& b,
                 std::vector<float>& out, bool relu) const {
        const int KS = in_ch * 9;
        thread_local std::vector<float> cols;
        cols.assign(static_cast<std::size_t>(KS) * nn, 0.0f);

        // im2col: scatter each (input-channel, kernel-tap) into its own row.
        for (int ic = 0; ic < in_ch; ++ic) {
            const float* inp = &in[static_cast<std::size_t>(ic) * nn];
            for (int kr = 0; kr < 3; ++kr) {
                int dr = kr - 1;
                for (int kc = 0; kc < 3; ++kc) {
                    int dc = kc - 1;
                    float* dst = &cols[static_cast<std::size_t>(ic * 9 + kr * 3 + kc) * nn];
                    for (int r = 0; r < n; ++r) {
                        int rr = r + dr;
                        if (rr < 0 || rr >= n) continue;
                        for (int c = 0; c < n; ++c) {
                            int cc = c + dc;
                            if (cc < 0 || cc >= n) continue;
                            dst[r * n + c] = inp[rr * n + cc];
                        }
                    }
                }
            }
        }

        out.assign(static_cast<std::size_t>(out_ch) * nn, 0.0f);
#if defined(__AVX2__)
        // AVX2 micro-kernel: output channels blocked 4 at a time, cells tiled by
        // 8 (one __m256). The 4 accumulators stay in registers across the whole
        // r loop, so each cols tile is loaded once and reused for 4 channels.
        // Uses FMA, so results differ from the scalar path only by FP rounding.
        {
            const __m256 zero = _mm256_setzero_ps();
            int oc = 0;
            for (; oc + 4 <= out_ch; oc += 4) {
                const float* wr0 = &w[static_cast<std::size_t>(oc + 0) * KS];
                const float* wr1 = &w[static_cast<std::size_t>(oc + 1) * KS];
                const float* wr2 = &w[static_cast<std::size_t>(oc + 2) * KS];
                const float* wr3 = &w[static_cast<std::size_t>(oc + 3) * KS];
                float* o0 = &out[static_cast<std::size_t>(oc + 0) * nn];
                float* o1 = &out[static_cast<std::size_t>(oc + 1) * nn];
                float* o2 = &out[static_cast<std::size_t>(oc + 2) * nn];
                float* o3 = &out[static_cast<std::size_t>(oc + 3) * nn];
                int cell = 0;
                for (; cell + 8 <= nn; cell += 8) {
                    __m256 a0 = _mm256_set1_ps(b[oc + 0]);
                    __m256 a1 = _mm256_set1_ps(b[oc + 1]);
                    __m256 a2 = _mm256_set1_ps(b[oc + 2]);
                    __m256 a3 = _mm256_set1_ps(b[oc + 3]);
                    for (int r = 0; r < KS; ++r) {
                        __m256 cv = _mm256_loadu_ps(&cols[static_cast<std::size_t>(r) * nn + cell]);
                        a0 = _mm256_fmadd_ps(_mm256_set1_ps(wr0[r]), cv, a0);
                        a1 = _mm256_fmadd_ps(_mm256_set1_ps(wr1[r]), cv, a1);
                        a2 = _mm256_fmadd_ps(_mm256_set1_ps(wr2[r]), cv, a2);
                        a3 = _mm256_fmadd_ps(_mm256_set1_ps(wr3[r]), cv, a3);
                    }
                    if (relu) {
                        a0 = _mm256_max_ps(a0, zero); a1 = _mm256_max_ps(a1, zero);
                        a2 = _mm256_max_ps(a2, zero); a3 = _mm256_max_ps(a3, zero);
                    }
                    _mm256_storeu_ps(&o0[cell], a0); _mm256_storeu_ps(&o1[cell], a1);
                    _mm256_storeu_ps(&o2[cell], a2); _mm256_storeu_ps(&o3[cell], a3);
                }
                for (; cell < nn; ++cell) {
                    float s0 = b[oc + 0], s1 = b[oc + 1], s2 = b[oc + 2], s3 = b[oc + 3];
                    for (int r = 0; r < KS; ++r) {
                        float cv = cols[static_cast<std::size_t>(r) * nn + cell];
                        s0 += wr0[r] * cv; s1 += wr1[r] * cv; s2 += wr2[r] * cv; s3 += wr3[r] * cv;
                    }
                    if (relu) { s0 = s0 < 0 ? 0 : s0; s1 = s1 < 0 ? 0 : s1; s2 = s2 < 0 ? 0 : s2; s3 = s3 < 0 ? 0 : s3; }
                    o0[cell] = s0; o1[cell] = s1; o2[cell] = s2; o3[cell] = s3;
                }
            }
            for (; oc < out_ch; ++oc) {
                const float* wrow = &w[static_cast<std::size_t>(oc) * KS];
                float* o = &out[static_cast<std::size_t>(oc) * nn];
                int cell = 0;
                for (; cell + 8 <= nn; cell += 8) {
                    __m256 a = _mm256_set1_ps(b[oc]);
                    for (int r = 0; r < KS; ++r)
                        a = _mm256_fmadd_ps(_mm256_set1_ps(wrow[r]),
                                            _mm256_loadu_ps(&cols[static_cast<std::size_t>(r) * nn + cell]), a);
                    if (relu) a = _mm256_max_ps(a, zero);
                    _mm256_storeu_ps(&o[cell], a);
                }
                for (; cell < nn; ++cell) {
                    float s = b[oc];
                    for (int r = 0; r < KS; ++r) s += wrow[r] * cols[static_cast<std::size_t>(r) * nn + cell];
                    if (relu && s < 0.0f) s = 0.0f;
                    o[cell] = s;
                }
            }
        }
#else
        // Scalar fallback (also used for non-AVX2 builds). Output channels are
        // blocked 4 at a time so each cols row is reused across 4 weights.
        int oc = 0;
        for (; oc + 4 <= out_ch; oc += 4) {
            float* __restrict o0 = &out[static_cast<std::size_t>(oc + 0) * nn];
            float* __restrict o1 = &out[static_cast<std::size_t>(oc + 1) * nn];
            float* __restrict o2 = &out[static_cast<std::size_t>(oc + 2) * nn];
            float* __restrict o3 = &out[static_cast<std::size_t>(oc + 3) * nn];
            const float* wr0 = &w[static_cast<std::size_t>(oc + 0) * KS];
            const float* wr1 = &w[static_cast<std::size_t>(oc + 1) * KS];
            const float* wr2 = &w[static_cast<std::size_t>(oc + 2) * KS];
            const float* wr3 = &w[static_cast<std::size_t>(oc + 3) * KS];
            for (int cell = 0; cell < nn; ++cell) {
                o0[cell] = b[oc + 0]; o1[cell] = b[oc + 1];
                o2[cell] = b[oc + 2]; o3[cell] = b[oc + 3];
            }
            for (int r = 0; r < KS; ++r) {
                const float* __restrict col = &cols[static_cast<std::size_t>(r) * nn];
                const float wv0 = wr0[r], wv1 = wr1[r], wv2 = wr2[r], wv3 = wr3[r];
                for (int cell = 0; cell < nn; ++cell) {
                    const float cv = col[cell];
                    o0[cell] += wv0 * cv; o1[cell] += wv1 * cv;
                    o2[cell] += wv2 * cv; o3[cell] += wv3 * cv;
                }
            }
            if (relu) {
                for (int j = 0; j < 4; ++j) {
                    float* o = &out[static_cast<std::size_t>(oc + j) * nn];
                    for (int cell = 0; cell < nn; ++cell) if (o[cell] < 0.0f) o[cell] = 0.0f;
                }
            }
        }
        for (; oc < out_ch; ++oc) {
            float* __restrict o = &out[static_cast<std::size_t>(oc) * nn];
            const float* wrow = &w[static_cast<std::size_t>(oc) * KS];
            for (int cell = 0; cell < nn; ++cell) o[cell] = b[oc];
            for (int r = 0; r < KS; ++r) {
                const float wv = wrow[r];
                const float* __restrict col = &cols[static_cast<std::size_t>(r) * nn];
                for (int cell = 0; cell < nn; ++cell) o[cell] += wv * col[cell];
            }
            if (relu) {
                for (int cell = 0; cell < nn; ++cell) if (o[cell] < 0.0f) o[cell] = 0.0f;
            }
        }
#endif
    }

    // 1x1 conv: out[oc*nn+cell] = b[oc] + sum_ic w[oc*in_ch+ic]*in[ic*nn+cell]
    void conv1x1(const std::vector<float>& in, int in_ch, int out_ch,
                 const std::vector<float>& w, const std::vector<float>& b,
                 std::vector<float>& out, bool relu) const {
        out.assign(out_ch * nn, 0.0f);
        for (int oc = 0; oc < out_ch; ++oc) {
            for (int cell = 0; cell < nn; ++cell) {
                float acc = b[oc];
                for (int ic = 0; ic < in_ch; ++ic) acc += w[oc * in_ch + ic] * in[ic * nn + cell];
                if (relu && acc < 0.0f) acc = 0.0f;
                out[oc * nn + cell] = acc;
            }
        }
    }

    Eval evaluate(const std::vector<int>& board, int player) const override {
        Eval e;
        e.logits.assign(nn, 0.0);

        std::vector<float> planes;
        make_planes(board, player, planes);

        std::vector<float> trunk;
        conv3x3(planes, in_planes, C, w_in, b_in, trunk, true);

        std::vector<float> h, h2;
        for (int blk = 0; blk < B; ++blk) {
            conv3x3(trunk, C, C, w1[blk], bb1[blk], h, true);
            conv3x3(h, C, C, w2[blk], bb2[blk], h2, false);
            for (int i = 0; i < C * nn; ++i) {
                float v = trunk[i] + h2[i];
                trunk[i] = v < 0.0f ? 0.0f : v;
            }
        }

        // policy head
        std::vector<float> p2;
        conv1x1(trunk, C, 2, w_ph, b_ph, p2, true);  // [2*nn]
        for (int i = 0; i < nn; ++i) {
            float acc = b_pf[i];
            const float* wr = &w_pf[static_cast<std::size_t>(i) * (2 * nn)];
            for (int k = 0; k < 2 * nn; ++k) acc += wr[k] * p2[k];
            e.logits[i] = acc;
        }

        // value head
        std::vector<float> v1;
        conv1x1(trunk, C, 1, w_vh, b_vh, v1, true);  // [nn]
        std::vector<float> vh(val_hidden, 0.0f);
        for (int j = 0; j < val_hidden; ++j) {
            float acc = b_vf1[j];
            const float* wr = &w_vf1[static_cast<std::size_t>(j) * nn];
            for (int k = 0; k < nn; ++k) acc += wr[k] * v1[k];
            vh[j] = acc < 0.0f ? 0.0f : acc;
        }
        float vraw = b_vf2;
        for (int j = 0; j < val_hidden; ++j) vraw += w_vf2[j] * vh[j];
        e.value = std::tanh(vraw);
        return e;
    }

    void init(int board_size, int channels, int blocks, uint64_t seed) {
        n = board_size;
        nn = n * n;
        C = channels;
        B = blocks;
        in_planes = 3;
        val_hidden = 64;
        Random rr(seed);
        auto fill = [&](std::vector<float>& v, std::size_t size, double scale) {
            v.assign(size, 0.0f);
            for (auto& x : v) x = static_cast<float>(rr.normal(0.0, scale));
        };
        auto zeros = [&](std::vector<float>& v, std::size_t size) { v.assign(size, 0.0f); };
        fill(w_in, static_cast<std::size_t>(C) * in_planes * 9, std::sqrt(2.0 / (in_planes * 9)));
        zeros(b_in, C);
        w1.assign(B, {}); bb1.assign(B, {}); w2.assign(B, {}); bb2.assign(B, {});
        double s_c = std::sqrt(2.0 / (C * 9));
        for (int blk = 0; blk < B; ++blk) {
            fill(w1[blk], static_cast<std::size_t>(C) * C * 9, s_c); zeros(bb1[blk], C);
            fill(w2[blk], static_cast<std::size_t>(C) * C * 9, s_c); zeros(bb2[blk], C);
        }
        fill(w_ph, static_cast<std::size_t>(2) * C, std::sqrt(2.0 / C)); zeros(b_ph, 2);
        fill(w_pf, static_cast<std::size_t>(nn) * (2 * nn), std::sqrt(2.0 / (2 * nn))); zeros(b_pf, nn);
        fill(w_vh, C, std::sqrt(2.0 / C)); zeros(b_vh, 1);
        fill(w_vf1, static_cast<std::size_t>(val_hidden) * nn, std::sqrt(2.0 / nn)); zeros(b_vf1, val_hidden);
        fill(w_vf2, val_hidden, std::sqrt(2.0 / val_hidden)); b_vf2 = 0.0f;
        ok = true;
    }

    bool save(const std::string& path) const {
        std::ofstream out(path);
        if (!out) return false;
        out << "HEXCNN_V1 " << n << ' ' << C << ' ' << B << ' ' << in_planes << ' ' << val_hidden << "\n";
        out << std::setprecision(9);
        auto wv = [&](const std::vector<float>& v) {
            out << v.size();
            for (float x : v) out << ' ' << x;
            out << "\n";
        };
        wv(w_in); wv(b_in);
        for (int blk = 0; blk < B; ++blk) { wv(w1[blk]); wv(bb1[blk]); wv(w2[blk]); wv(bb2[blk]); }
        wv(w_ph); wv(b_ph); wv(w_pf); wv(b_pf);
        wv(w_vh); wv(b_vh); wv(w_vf1); wv(b_vf1); wv(w_vf2);
        out << b_vf2 << "\n";
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream in(path);
        if (!in) return false;
        std::string tag;
        in >> tag;
        if (tag != "HEXCNN_V1") return false;
        in >> n >> C >> B >> in_planes >> val_hidden;
        nn = n * n;
        auto rv = [&](std::vector<float>& v) {
            std::size_t s = 0;
            in >> s;
            v.assign(s, 0.0f);
            for (auto& x : v) { double d = 0.0; in >> d; x = static_cast<float>(d); }
        };
        rv(w_in); rv(b_in);
        w1.assign(B, {}); bb1.assign(B, {}); w2.assign(B, {}); bb2.assign(B, {});
        for (int blk = 0; blk < B; ++blk) { rv(w1[blk]); rv(bb1[blk]); rv(w2[blk]); rv(bb2[blk]); }
        rv(w_ph); rv(b_ph); rv(w_pf); rv(b_pf);
        rv(w_vh); rv(b_vh); rv(w_vf1); rv(b_vf1); rv(w_vf2);
        double d = 0.0; in >> d; b_vf2 = static_cast<float>(d);
        ok = static_cast<int>(w_in.size()) == C * in_planes * 9 &&
             static_cast<std::size_t>(w_pf.size()) == static_cast<std::size_t>(nn) * (2 * nn) &&
             static_cast<std::size_t>(w_vf1.size()) == static_cast<std::size_t>(val_hidden) * nn &&
             static_cast<int>(w_vf2.size()) == val_hidden;
        return ok;
    }
};

struct TrainingExample {
    int n = 9;
    int player = BLACK;
    std::vector<int> board;
    std::vector<double> policy;
    int winner = EMPTY;
    double value = 0.0;
};

std::string serialize_policy(const std::vector<double>& policy) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8);
    bool first = true;
    for (int i = 0; i < static_cast<int>(policy.size()); ++i) {
        if (policy[i] <= 0.0) continue;
        if (!first) oss << ',';
        first = false;
        oss << i << ':' << policy[i];
    }
    return oss.str();
}

std::vector<double> parse_policy(const std::string& text, int nn) {
    std::vector<double> policy(nn, 0.0);
    for (const std::string& item : split(text, ',')) {
        if (item.empty()) continue;
        auto pos = item.find(':');
        if (pos == std::string::npos) continue;
        int idx = std::stoi(item.substr(0, pos));
        double p = std::stod(item.substr(pos + 1));
        if (idx >= 0 && idx < nn) policy[idx] = p;
    }
    double sum = std::accumulate(policy.begin(), policy.end(), 0.0);
    if (sum > 0.0) {
        for (double& p : policy) p /= sum;
    }
    return policy;
}

bool append_examples(const std::string& path, const std::vector<TrainingExample>& examples) {
    bool need_header = true;
    {
        std::ifstream in(path);
        need_header = !in.good() || in.peek() == std::ifstream::traits_type::eof();
    }
    std::ofstream out(path, std::ios::app);
    if (!out) return false;
    if (need_header) {
        out << "# HEXSELFPLAY_V1\n";
        out << "# n\tplayer\tboard\tpolicy\twinner\tvalue\n";
    }
    out << std::fixed << std::setprecision(8);
    for (const auto& ex : examples) {
        out << ex.n << '\t'
            << ex.player << '\t'
            << board_to_string(ex.board) << '\t'
            << serialize_policy(ex.policy) << '\t'
            << ex.winner << '\t'
            << ex.value << '\n';
    }
    return true;
}

std::vector<TrainingExample> load_examples(const std::string& path, int expected_n, int limit = 0) {
    std::ifstream in(path);
    std::vector<TrainingExample> data;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto cols = split(line, '\t');
        if (cols.size() < 6) continue;
        TrainingExample ex;
        ex.n = std::stoi(cols[0]);
        if (expected_n > 0 && ex.n != expected_n) continue;
        int nn = ex.n * ex.n;
        ex.player = std::stoi(cols[1]);
        if (!parse_board_text(cols[2], nn, ex.board)) continue;
        ex.policy = parse_policy(cols[3], nn);
        ex.winner = std::stoi(cols[4]);
        ex.value = std::stod(cols[5]);
        data.push_back(std::move(ex));
        if (limit > 0 && static_cast<int>(data.size()) >= limit) break;
    }
    return data;
}

std::vector<double> heuristic_priors(const GameState& gs) {
    std::vector<double> priors(gs.rules->nn, 0.0);
    auto moves = legal_moves(gs);
    if (moves.empty()) return priors;

    std::vector<int> winning;
    std::vector<int> blocks;
    find_critical_cells(gs, winning, blocks);
    std::set<int> win_set(winning.begin(), winning.end());
    std::set<int> block_set(blocks.begin(), blocks.end());
    auto saves = get_bridge_saves(*gs.rules, gs.board, gs.cur, gs.last_move);
    std::set<int> save_set(saves.begin(), saves.end());

    double center = (gs.rules->n - 1) / 2.0;
    double sum = 0.0;
    for (int m : moves) {
        double p = 1.0;
        if (win_set.count(m)) p += 200.0;
        if (block_set.count(m)) p += 80.0;
        if (save_set.count(m)) p += 10.0;
        p += std::max(0.0, 5.0 - (std::abs(m / gs.rules->n - center) + std::abs(m % gs.rules->n - center)));
        priors[m] = p;
        sum += p;
    }
    if (sum > 0.0) {
        for (double& p : priors) p /= sum;
    }
    return priors;
}

// One network evaluation per node. Both the move priors and the node's value
// are derived from a single forward pass. Previously a node ran the (identical)
// inference twice: once in the node constructor for the priors and again in the
// leaf evaluation for the value. The result is byte-for-byte identical because
// the network is deterministic and consumes no RNG.
struct NodeEval {
    std::vector<double> priors;
    double net_black = 0.5;  // P(Black wins) from the network, if available
    bool has_net = false;
};

NodeEval eval_node(const GameState& gs, const NeuralNet* net) {
    NodeEval e;
    if (!net || !net->is_ready() || net->board_n() != gs.rules->n) {
        e.priors = heuristic_priors(gs);
        return e;
    }
    Eval f = net->evaluate(gs.board, gs.cur);

    std::vector<double> priors(gs.rules->nn, 0.0);
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < gs.rules->nn; ++i) {
        if (gs.board[i] == EMPTY) max_logit = std::max(max_logit, f.logits[i]);
    }
    double sum = 0.0;
    for (int i = 0; i < gs.rules->nn; ++i) {
        if (gs.board[i] != EMPTY) continue;
        priors[i] = std::exp(f.logits[i] - max_logit);
        sum += priors[i];
    }
    if (sum <= 0.0 || !std::isfinite(sum)) {
        e.priors = heuristic_priors(gs);
    } else {
        for (double& p : priors) p /= sum;
        e.priors = std::move(priors);
    }

    double current_win_prob = (f.value + 1.0) * 0.5;
    e.net_black = gs.cur == BLACK ? current_win_prob : (1.0 - current_win_prob);
    e.has_net = true;
    return e;
}

struct SearchConfig {
    int iterations = 1500;
    int endgame_depth = 5;
    double exploration = 1.15;
    double rollout_weight = 0.45;
    bool use_rollout = true;
    // Self-play exploration: Dirichlet noise mixed into the root priors.
    bool add_root_noise = false;
    double dirichlet_alpha = 0.3;
    double dirichlet_epsilon = 0.25;
    // First-play urgency: optimism applied to not-yet-expanded moves so PUCT
    // can lean on the policy network instead of expanding every sibling first.
    double fpu_reduction = 0.25;
    // MCTS-Solver: back up proven win/loss results to play forced lines exactly.
    bool use_solver = true;
    // Blend the root policy prior with the hand heuristic (0 = pure net prior,
    // 1 = pure heuristic). A weak/biased net opening prior can be corrected by
    // mixing in the heuristic's centre/connection preference.
    double prior_blend = 0.0;
};

struct SearchResult {
    int move = -1;
    std::vector<int> visits;
    double black_win_estimate = 0.5;
};

struct MCTSNode {
    GameState st;
    MCTSNode* parent = nullptr;
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<int> untried;
    std::vector<double> priors;
    int visits = 0;
    double black_wins = 0.0;
    int move = -1;
    double prior_from_parent = 1.0;
    double cached_net_black = 0.5;  // network value cached from construction
    bool has_cached_net = false;
    // MCTS-Solver: proven game-theoretic value from st.cur's perspective.
    //   +1 = side to move wins with best play, -1 = loses, 0 = unknown.
    int proven = 0;

    MCTSNode(const GameState& state, MCTSNode* par, int mv, double prior, const NeuralNet* net)
        : st(state), parent(par), move(mv), prior_from_parent(prior) {
        NodeEval e = eval_node(st, net);
        priors = std::move(e.priors);
        cached_net_black = e.net_black;
        has_cached_net = e.has_net;
        untried = legal_moves(st);
        // A terminal position means the previous mover just connected, so the
        // side to move here has already lost.
        if (st.is_terminal()) proven = -1;
    }

    // Win probability at this node from the perspective of the side to move.
    double value_for_cur() const {
        if (visits <= 0) return 0.5;
        double black_q = black_wins / visits;
        return st.cur == BLACK ? black_q : (1.0 - black_q);
    }
};

// Leaf value from Black's perspective. The network term reuses the value that
// was already computed (and cached on the node) during expansion, so no second
// forward pass is needed.
double evaluate_leaf_black_win(const GameState& st, double cached_net_black, bool has_net, const SearchConfig& config) {
    if (st.has_won(BLACK)) return 1.0;
    if (st.has_won(WHITE)) return 0.0;

    double heuristic = heuristic_black_win(st);
    double value = heuristic;
    if (config.use_rollout) {
        double rollout = rollout_light(st);
        value = (1.0 - config.rollout_weight) * heuristic + config.rollout_weight * rollout;
    }
    if (has_net) {
        value = 0.50 * cached_net_black + 0.30 * value + 0.20 * heuristic;
    }
    return std::max(0.0, std::min(1.0, value));
}

// Mix Dirichlet noise into the priors of the given moves (root exploration).
void add_dirichlet_noise(std::vector<double>& priors,
                         const std::vector<int>& moves,
                         double alpha,
                         double eps) {
    if (moves.empty() || eps <= 0.0) return;
    std::vector<double> noise(moves.size(), 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < moves.size(); ++i) {
        double g = g_rng.gamma(alpha);
        if (!(g > 0.0)) g = 1e-9;
        noise[i] = g;
        sum += g;
    }
    if (sum <= 0.0) return;
    for (std::size_t i = 0; i < moves.size(); ++i) {
        double nz = noise[i] / sum;
        int m = moves[i];
        priors[m] = (1.0 - eps) * priors[m] + eps * nz;
    }
}

SearchResult mcts_search(const GameState& root_state, const SearchConfig& config, const NeuralNet* net) {
    SearchResult result;
    result.visits.assign(root_state.rules->nn, 0);

    std::vector<int> winning;
    std::vector<int> must_block;
    find_critical_cells(root_state, winning, must_block);
    if (!winning.empty()) {
        result.move = winning.front();
        result.visits[result.move] = 1;
        result.black_win_estimate = root_state.cur == BLACK ? 1.0 : 0.0;
        return result;
    }
    if (must_block.size() == 1) {
        result.move = must_block.front();
        result.visits[result.move] = 1;
        return result;
    }

    MCTSNode root(root_state, nullptr, -1, 1.0, net);
    if (root.untried.empty()) return result;
    if (config.prior_blend > 0.0) {
        std::vector<double> heur = heuristic_priors(root_state);
        double b = std::min(1.0, config.prior_blend);
        for (int i = 0; i < root_state.rules->nn; ++i) {
            root.priors[i] = (1.0 - b) * root.priors[i] + b * heur[i];
        }
    }
    if (config.add_root_noise) {
        add_dirichlet_noise(root.priors, root.untried, config.dirichlet_alpha, config.dirichlet_epsilon);
    }

    for (int iter = 0; iter < config.iterations; ++iter) {
        MCTSNode* node = &root;
        std::vector<MCTSNode*> path;
        path.push_back(node);

        // Selection: at each node compare the best already-expanded child
        // against the best not-yet-expanded move (using first-play urgency),
        // so promising lines can be deepened without first expanding every
        // sibling. With a trained policy the priors are sharp enough that this
        // focuses the limited simulation budget on strong moves.
        // Stop descending at terminal or already-proven nodes (treat as leaves).
        while (!node->st.is_terminal() && !(config.use_solver && node->proven != 0)) {
            double sqrt_n = std::sqrt(static_cast<double>(std::max(1, node->visits)));
            double fpu = std::max(0.0, std::min(1.0, node->value_for_cur() - config.fpu_reduction));

            int best_untried = -1;
            double best_untried_score = -std::numeric_limits<double>::infinity();
            for (int m : node->untried) {
                double prior = node->priors[m];
                double score = fpu + config.exploration * prior * sqrt_n;
                if (score > best_untried_score) {
                    best_untried_score = score;
                    best_untried = m;
                }
            }

            MCTSNode* best_child = nullptr;
            double best_child_score = -std::numeric_limits<double>::infinity();
            for (const auto& child : node->children) {
                double score;
                if (config.use_solver && child->proven == -1) {
                    score = std::numeric_limits<double>::infinity();    // move wins for us
                } else if (config.use_solver && child->proven == 1) {
                    score = -std::numeric_limits<double>::infinity();   // move loses for us
                } else {
                    double q = fpu;
                    if (child->visits > 0) {
                        double black_q = child->black_wins / child->visits;
                        q = node->st.cur == BLACK ? black_q : (1.0 - black_q);
                    }
                    score = q + config.exploration * child->prior_from_parent * sqrt_n / (1.0 + child->visits);
                }
                if (score > best_child_score) {
                    best_child_score = score;
                    best_child = child.get();
                }
            }

            bool expand;
            if (!best_child) {
                expand = best_untried >= 0;
            } else if (best_untried < 0) {
                expand = false;
            } else {
                expand = best_untried_score >= best_child_score;
            }

            if (expand) {
                int m = best_untried;
                auto it = std::find(node->untried.begin(), node->untried.end(), m);
                if (it != node->untried.end()) {
                    *it = node->untried.back();
                    node->untried.pop_back();
                }
                GameState ns = node->st.apply(m);
                double prior = node->priors[m] > 0.0 ? node->priors[m] : 1.0 / std::max(1, node->st.empty_count());
                node->children.push_back(std::make_unique<MCTSNode>(ns, node, m, prior, net));
                node = node->children.back().get();
                path.push_back(node);
                break;  // newly expanded leaf is evaluated below
            }

            node = best_child;
            path.push_back(node);
        }

        double black_win;
        if (config.use_solver && node->proven != 0) {
            double cur_win = (node->proven == 1) ? 1.0 : 0.0;
            black_win = (node->st.cur == BLACK) ? cur_win : (1.0 - cur_win);
        } else {
            black_win = evaluate_leaf_black_win(node->st, node->cached_net_black, node->has_cached_net, config);
        }
        for (MCTSNode* p : path) {
            ++p->visits;
            p->black_wins += black_win;
        }

        if (config.use_solver) {
            // MCTS-Solver: back up proven win/loss along the visited path.
            for (int i = static_cast<int>(path.size()) - 2; i >= 0; --i) {
                MCTSNode* par = path[i];
                if (par->proven != 0) break;
                const MCTSNode* ch = path[i + 1];
                if (ch->proven == -1) {
                    par->proven = 1;  // a move into the opponent's loss is our win
                } else if (ch->proven == 1) {
                    if (!par->untried.empty()) break;  // unexplored moves remain
                    bool all_lost = true;
                    for (const auto& c : par->children) {
                        if (c->proven != 1) { all_lost = false; break; }
                    }
                    if (all_lost) par->proven = -1; else break;
                } else {
                    break;  // child value still unknown
                }
            }
            if (root.proven != 0) break;  // root is solved; no need to search further
        }
    }

    // Final move choice (MCTS-Solver aware):
    //   1. a proven winning move if one exists,
    //   2. otherwise the most-visited move that is not a proven loss,
    //   3. otherwise the most-visited move (position is lost either way).
    MCTSNode* best = nullptr;          // most visits overall
    MCTSNode* best_win = nullptr;      // proven win for us (child proven == -1)
    MCTSNode* best_nonloss = nullptr;  // most visits among non-losing moves
    for (const auto& child : root.children) {
        result.visits[child->move] = child->visits;
        if (!best || child->visits > best->visits) best = child.get();
        if (config.use_solver) {
            if (child->proven == -1 && (!best_win || child->visits > best_win->visits)) best_win = child.get();
            if (child->proven != 1 && (!best_nonloss || child->visits > best_nonloss->visits)) best_nonloss = child.get();
        }
    }
    MCTSNode* chosen = config.use_solver ? (best_win ? best_win : (best_nonloss ? best_nonloss : best)) : best;
    if (chosen) {
        result.move = chosen->move;
        result.black_win_estimate = chosen->visits > 0 ? chosen->black_wins / chosen->visits : 0.5;
        if (config.use_solver && best_win) {
            // Make the visit-based policy target (used by self-play) agree with
            // the proven winning move.
            int maxv = 0;
            for (const auto& child : root.children) maxv = std::max(maxv, child->visits);
            result.visits[chosen->move] = std::max(result.visits[chosen->move], maxv + 1);
        }
    } else {
        auto moves = legal_moves(root_state);
        result.move = moves.empty() ? -1 : moves.front();
    }
    return result;
}

SearchResult search_position(const GameState& gs, const SearchConfig& config, const NeuralNet* net) {
    if (config.endgame_depth > 0 && gs.empty_count() <= config.endgame_depth) {
        SearchResult result;
        result.visits.assign(gs.rules->nn, 0);
        result.move = alphabeta_best_move(gs, gs.empty_count());
        if (result.move >= 0) result.visits[result.move] = 1;
        return result;
    }
    return mcts_search(gs, config, net);
}

std::vector<double> visits_to_policy(const std::vector<int>& visits, int fallback_move) {
    std::vector<double> policy(visits.size(), 0.0);
    int sum = std::accumulate(visits.begin(), visits.end(), 0);
    if (sum > 0) {
        for (int i = 0; i < static_cast<int>(visits.size()); ++i) policy[i] = static_cast<double>(visits[i]) / sum;
    } else if (fallback_move >= 0 && fallback_move < static_cast<int>(policy.size())) {
        policy[fallback_move] = 1.0;
    }
    return policy;
}

int sample_from_visits(const std::vector<int>& visits, const GameState& st, double temperature, int fallback) {
    auto moves = legal_moves(st);
    if (moves.empty()) return -1;
    if (temperature <= 1e-9) {
        int best = fallback >= 0 ? fallback : moves.front();
        int best_v = -1;
        for (int m : moves) {
            if (m < static_cast<int>(visits.size()) && visits[m] > best_v) {
                best_v = visits[m];
                best = m;
            }
        }
        return best;
    }

    std::vector<double> weights;
    weights.reserve(moves.size());
    double sum = 0.0;
    for (int m : moves) {
        double v = m < static_cast<int>(visits.size()) ? static_cast<double>(visits[m]) : 0.0;
        double w = std::pow(std::max(0.0, v), 1.0 / temperature);
        if (w <= 0.0) w = 1e-6;
        weights.push_back(w);
        sum += w;
    }
    double r = g_rng.uniform01() * sum;
    double acc = 0.0;
    for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
        acc += weights[i];
        if (r <= acc) return moves[i];
    }
    return moves.back();
}

struct Args {
    std::string command = "play";
    std::unordered_map<std::string, std::string> opt;
};

Args parse_args(int argc, char* argv[]) {
    Args args;
    if (argc > 1) args.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) == 0) {
            key = key.substr(2);
            std::string value = "1";
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) value = argv[++i];
            args.opt[key] = value;
        }
    }
    return args;
}

int get_int(const Args& args, const std::string& key, int fallback) {
    auto it = args.opt.find(key);
    if (it == args.opt.end()) return fallback;
    return std::stoi(it->second);
}

double get_double(const Args& args, const std::string& key, double fallback) {
    auto it = args.opt.find(key);
    if (it == args.opt.end()) return fallback;
    return std::stod(it->second);
}

std::string get_string(const Args& args, const std::string& key, const std::string& fallback) {
    auto it = args.opt.find(key);
    if (it == args.opt.end()) return fallback;
    return it->second;
}

SearchConfig config_from_args(const Args& args, int n) {
    SearchConfig cfg;
    cfg.iterations = get_int(args, "iters", n <= 7 ? 2500 : (n <= 9 ? 2300 : 900));
    cfg.endgame_depth = get_int(args, "endgame", n <= 7 ? 8 : (n <= 9 ? 5 : 3));
    cfg.exploration = get_double(args, "cpuct", 1.15);
    cfg.rollout_weight = get_double(args, "rollout-weight", 0.45);
    cfg.use_rollout = get_int(args, "rollout", 1) != 0;
    cfg.dirichlet_alpha = get_double(args, "dir-alpha", 0.3);
    cfg.dirichlet_epsilon = get_double(args, "dir-eps", 0.25);
    cfg.fpu_reduction = get_double(args, "fpu", 0.25);
    cfg.use_solver = get_int(args, "solver", 1) != 0;
    cfg.prior_blend = get_double(args, "prior-blend", 0.0);
    // Root noise stays off for interactive play / analysis; self-play turns it
    // on explicitly so generated games keep exploring openings.
    cfg.add_root_noise = get_int(args, "root-noise", 0) != 0;
    return cfg;
}

// Load either model format, auto-detected from the file header tag.
std::unique_ptr<NeuralNet> load_any_model(const std::string& path, int n) {
    if (path.empty()) return nullptr;
    std::string tag;
    {
        std::ifstream in(path);
        if (!in) {
            std::cerr << "model open failed: " << path << "\n";
            return nullptr;
        }
        in >> tag;
    }
    if (tag == "HEXCNN_V1") {
        auto net = std::make_unique<ConvNet>();
        if (!net->load(path)) {
            std::cerr << "cnn load failed: " << path << "\n";
            return nullptr;
        }
        if (net->n != n) {
            std::cerr << "cnn board size mismatch. model=" << net->n << " requested=" << n << "\n";
            return nullptr;
        }
        return net;
    }
    auto net = std::make_unique<TinyNet>();
    if (!net->load(path)) {
        std::cerr << "model load failed: " << path << "\n";
        return nullptr;
    }
    if (net->n != n) {
        std::cerr << "model board size mismatch. model=" << net->n << " requested=" << n << "\n";
        return nullptr;
    }
    return net;
}

std::unique_ptr<NeuralNet> load_model_if_requested(const Args& args, int n) {
    return load_any_model(get_string(args, "model", ""), n);
}

int run_play(const Args& args) {
    int n = get_int(args, "n", 9);
    g_rng = Random(static_cast<uint64_t>(get_int(args, "seed", 123456789)));
    auto rules = std::make_shared<Rules>(n);
    GameState gs(rules);
    SearchConfig cfg = config_from_args(args, n);
    auto net = load_model_if_requested(args, n);

    std::cout << "Hex AI " << n << "x" << n
              << "  MCTS=" << cfg.iterations
              << "  endgame=" << cfg.endgame_depth
              << (net ? "  model=on" : "  model=off") << "\n";
    std::cout << "Black connects top-bottom. White connects left-right.\n";

    int human = -1;
    std::string color_arg = get_string(args, "human", "");
    if (color_arg == "b" || color_arg == "B" || color_arg == "black") human = BLACK;
    if (color_arg == "w" || color_arg == "W" || color_arg == "white") human = WHITE;
    while (human < 0) {
        std::cout << "Your color (b/w): ";
        std::string s;
        std::cin >> s;
        if (s == "b" || s == "B") human = BLACK;
        else if (s == "w" || s == "W") human = WHITE;
    }

    print_board(gs);
    while (!gs.is_terminal()) {
        if (gs.cur == human) {
            while (true) {
                std::cout << "Your move: ";
                std::string raw;
                std::cin >> raw;
                int move = rules->move_from_string(raw);
                if (move < 0) {
                    std::cout << "Invalid coordinate.\n";
                    continue;
                }
                if (gs.board[move] != EMPTY) {
                    std::cout << "That cell is already occupied.\n";
                    continue;
                }
                gs = gs.apply(move);
                std::cout << "You -> " << rules->move_to_string(move) << "\n";
                print_board(gs);
                break;
            }
        } else {
            std::cout << "AI thinking..." << std::flush;
            SearchResult sr = search_position(gs, cfg, net.get());
            if (sr.move < 0) break;
            gs = gs.apply(sr.move);
            std::cout << "\rAI -> " << rules->move_to_string(sr.move) << "       \n";
            print_board(gs);
        }
    }

    int w = gs.winner();
    std::cout << (w == BLACK ? "Black" : "White") << " wins.\n";
    return 0;
}

int run_move(const Args& args) {
    int n = get_int(args, "n", 9);
    auto rules = std::make_shared<Rules>(n);
    GameState gs(rules);
    std::string board_text = get_string(args, "board", "");
    if (!parse_board_text(board_text, rules->nn, gs.board)) {
        std::cerr << "Use --board with " << rules->nn << " chars: . B W or 0 1 2.\n";
        return 2;
    }
    gs.cur = get_int(args, "player", BLACK);
    gs.last_move = get_int(args, "last", -1);
    rebuild_union_find(gs);
    SearchConfig cfg = config_from_args(args, n);
    auto net = load_model_if_requested(args, n);
    SearchResult sr = search_position(gs, cfg, net.get());
    std::cout << sr.move << " " << rules->move_to_string(sr.move) << "\n";
    return 0;
}

// Print the raw network output (value + all policy logits) for one position.
// No search. Used to cross-validate the Python trainer's export against the
// C++ reader: identical board/player/model must give identical numbers.
int run_eval(const Args& args) {
    int n = get_int(args, "n", 9);
    auto rules = std::make_shared<Rules>(n);
    GameState gs(rules);
    std::string board_text = get_string(args, "board", std::string(rules->nn, '.'));
    if (!parse_board_text(board_text, rules->nn, gs.board)) {
        std::cerr << "Use --board with " << rules->nn << " chars: . B W or 0 1 2.\n";
        return 2;
    }
    int player = get_int(args, "player", BLACK);
    auto net = load_model_if_requested(args, n);
    if (!net || !net->is_ready()) {
        std::cerr << "eval needs a valid --model\n";
        return 2;
    }
    Eval e = net->evaluate(gs.board, player);
    std::cout << std::setprecision(9);
    std::cout << "value " << e.value << "\n";
    std::cout << "logits";
    for (double l : e.logits) std::cout << ' ' << l;
    std::cout << "\n";
    return 0;
}

int run_selfplay(const Args& args) {
    int n = get_int(args, "n", 9);
    int games = get_int(args, "games", 10);
    int temp_moves = get_int(args, "temp-moves", n <= 7 ? 10 : 18);
    std::string out_path = get_string(args, "out", "hex_selfplay.tsv");
    g_rng = Random(static_cast<uint64_t>(get_int(args, "seed", 123456789)));
    auto rules = std::make_shared<Rules>(n);
    SearchConfig cfg = config_from_args(args, n);
    // Self-play explores openings via root noise unless explicitly disabled.
    cfg.add_root_noise = get_int(args, "root-noise", 1) != 0;
    auto net = load_model_if_requested(args, n);

    int total_positions = 0;
    for (int game = 1; game <= games; ++game) {
        GameState st(rules);
        std::vector<TrainingExample> history;
        int ply = 0;
        while (!st.is_terminal() && !legal_moves(st).empty()) {
            SearchResult sr = search_position(st, cfg, net.get());
            int chosen = sample_from_visits(sr.visits, st, ply < temp_moves ? 1.0 : 0.0, sr.move);
            if (chosen < 0) break;

            TrainingExample ex;
            ex.n = n;
            ex.player = st.cur;
            ex.board = st.board;
            ex.policy = visits_to_policy(sr.visits, chosen);
            history.push_back(std::move(ex));

            st = st.apply(chosen);
            ++ply;
        }
        int winner = st.winner();
        for (auto& ex : history) {
            ex.winner = winner;
            ex.value = (winner == ex.player) ? 1.0 : -1.0;
        }
        if (!append_examples(out_path, history)) {
            std::cerr << "failed to append: " << out_path << "\n";
            return 3;
        }
        total_positions += static_cast<int>(history.size());
        std::cout << "game " << game << "/" << games
                  << " winner=" << (winner == BLACK ? "B" : "W")
                  << " positions=" << history.size()
                  << " total=" << total_positions << "\n";
    }
    std::cout << "saved: " << out_path << "\n";
    return 0;
}

int run_train(const Args& args) {
    int n = get_int(args, "n", 9);
    int epochs = get_int(args, "epochs", 5);
    int h1 = get_int(args, "h1", 128);
    int h2 = get_int(args, "h2", 64);
    int limit = get_int(args, "limit", 0);
    double lr = get_double(args, "lr", 0.01);
    double value_weight = get_double(args, "value-weight", 0.35);
    std::string data_path = get_string(args, "data", "hex_selfplay.tsv");
    std::string model_in = get_string(args, "model-in", "");
    std::string model_out = get_string(args, "model-out", get_string(args, "model", "hex_model.nn"));

    auto data = load_examples(data_path, n, limit);
    if (data.empty()) {
        std::cerr << "no training examples loaded from " << data_path << "\n";
        return 4;
    }

    TinyNet net;
    if (!model_in.empty() && net.load(model_in)) {
        std::cout << "loaded model: " << model_in << "\n";
    } else {
        net.init(n, h1, h2, static_cast<uint64_t>(get_int(args, "seed", 20240531)));
        std::cout << "initialized model: n=" << n << " h1=" << h1 << " h2=" << h2 << "\n";
    }

    std::vector<int> order(data.size());
    std::iota(order.begin(), order.end(), 0);
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), g_rng.engine);
        double loss_sum = 0.0;
        for (int idx : order) {
            const auto& ex = data[idx];
            loss_sum += net.train_one(ex.board, ex.player, ex.policy, ex.value, lr, value_weight);
        }
        std::cout << "epoch " << epoch << "/" << epochs
                  << " loss=" << (loss_sum / std::max<std::size_t>(1, data.size())) << "\n";
        lr *= get_double(args, "lr-decay", 0.97);
    }

    if (!net.save(model_out)) {
        std::cerr << "model save failed: " << model_out << "\n";
        return 5;
    }
    std::cout << "saved model: " << model_out << "\n";
    return 0;
}

int run_pipeline(const Args& args) {
    int cycles = get_int(args, "cycles", 3);
    std::string data = get_string(args, "data", "hex_selfplay.tsv");
    std::string model = get_string(args, "model", "hex_model.nn");
    for (int c = 1; c <= cycles; ++c) {
        std::cout << "cycle " << c << "/" << cycles << ": selfplay\n";
        Args sp = args;
        sp.command = "selfplay";
        sp.opt["out"] = data;
        std::ifstream model_in(model);
        if (model_in.good()) sp.opt["model"] = model;
        int rc = run_selfplay(sp);
        if (rc != 0) return rc;

        std::cout << "cycle " << c << "/" << cycles << ": train\n";
        Args tr = args;
        tr.command = "train";
        tr.opt["data"] = data;
        tr.opt["model-out"] = model;
        std::ifstream model_in2(model);
        if (model_in2.good()) tr.opt["model-in"] = model;
        rc = run_train(tr);
        if (rc != 0) return rc;
    }
    return 0;
}

// Build a search config for one side of a match. Starts from the shared
// defaults and applies any per-side overrides carrying the given suffix
// (e.g. "iters-a", "rollout-b", "cpuct-a").
SearchConfig side_config(const Args& args, int n, const std::string& suffix) {
    SearchConfig cfg = config_from_args(args, n);
    cfg.add_root_noise = false;  // matches use deterministic best-move selection
    if (args.opt.count("iters" + suffix)) cfg.iterations = get_int(args, "iters" + suffix, cfg.iterations);
    if (args.opt.count("endgame" + suffix)) cfg.endgame_depth = get_int(args, "endgame" + suffix, cfg.endgame_depth);
    if (args.opt.count("cpuct" + suffix)) cfg.exploration = get_double(args, "cpuct" + suffix, cfg.exploration);
    if (args.opt.count("fpu" + suffix)) cfg.fpu_reduction = get_double(args, "fpu" + suffix, cfg.fpu_reduction);
    if (args.opt.count("rollout" + suffix)) cfg.use_rollout = get_int(args, "rollout" + suffix, 1) != 0;
    if (args.opt.count("rollout-weight" + suffix))
        cfg.rollout_weight = get_double(args, "rollout-weight" + suffix, cfg.rollout_weight);
    if (args.opt.count("solver" + suffix)) cfg.use_solver = get_int(args, "solver" + suffix, 1) != 0;
    if (args.opt.count("prior-blend" + suffix)) cfg.prior_blend = get_double(args, "prior-blend" + suffix, cfg.prior_blend);
    return cfg;
}

// Engine-vs-engine match for measuring strength. Side A and side B can use
// different models (--model-a/--model-b, falling back to --model) and different
// search settings (suffix -a/-b). Games are played in color-swapped pairs that
// share a short random opening (--opening-plies, default 4). The random opening
// diversifies the games and neutralises Hex's decisive first-player advantage,
// so a genuine strength gap shows up in the win rate instead of being masked by
// "whoever plays Black wins". Engine moves themselves stay deterministic
// (best move, no root noise) so the measurement is low variance, and everything
// is seeded from --seed so results are reproducible yet varied.
int run_match(const Args& args) {
    int n = get_int(args, "n", 9);
    int games = get_int(args, "games", 20);
    uint64_t base_seed = static_cast<uint64_t>(get_int(args, "seed", 12345));
    int opening_plies = get_int(args, "opening-plies", 4);
    auto rules = std::make_shared<Rules>(n);

    std::string model_shared = get_string(args, "model", "");
    std::string model_a = get_string(args, "model-a", model_shared);
    std::string model_b = get_string(args, "model-b", model_shared);
    auto net_a = load_any_model(model_a, n);
    auto net_b = load_any_model(model_b, n);

    SearchConfig cfg_a = side_config(args, n, "-a");
    SearchConfig cfg_b = side_config(args, n, "-b");
    bool verbose = get_int(args, "verbose", 0) != 0;

    int a_wins = 0, b_wins = 0;
    int a_black = 0, a_white = 0, b_black = 0, b_white = 0;
    int pairs = (games + 1) / 2;
    int played = 0;
    for (int pr = 0; pr < pairs; ++pr) {
        // One random opening per pair, replayed with colors swapped so any
        // residual opening bias cancels between the two games.
        std::vector<int> opening;
        Random orng(base_seed * 1000003ull + static_cast<uint64_t>(pr) + 1);
        {
            GameState os(rules);
            for (int k = 0; k < opening_plies; ++k) {
                std::vector<int> moves = legal_moves(os);
                if (os.is_terminal() || moves.empty()) break;
                int pick = moves[orng.uniform_int(0, static_cast<int>(moves.size()) - 1)];
                opening.push_back(pick);
                os = os.apply(pick);
            }
        }
        for (int side = 0; side < 2 && played < games; ++side) {
            ++played;
            bool a_is_black = (side == 0);
            g_rng = Random(base_seed + static_cast<uint64_t>(played));
            GameState st(rules);
            for (int mv : opening) st = st.apply(mv);
            while (!st.is_terminal() && !legal_moves(st).empty()) {
                bool a_to_move = (st.cur == BLACK) == a_is_black;
                const SearchConfig& cfg = a_to_move ? cfg_a : cfg_b;
                const NeuralNet* net = a_to_move ? net_a.get() : net_b.get();
                SearchResult sr = search_position(st, cfg, net);
                if (sr.move < 0) break;
                st = st.apply(sr.move);
            }
            int w = st.winner();
            bool black_won = (w == BLACK);
            bool a_won = (black_won == a_is_black);
            if (a_won) {
                ++a_wins;
                if (a_is_black) ++a_black; else ++a_white;
            } else {
                ++b_wins;
                if (a_is_black) ++b_white; else ++b_black;
            }
            if (verbose) {
                std::cout << "game " << played << "/" << games
                          << " A=" << (a_is_black ? "Black" : "White")
                          << " winner=" << (black_won ? "Black" : "White")
                          << " -> " << (a_won ? "A" : "B") << "\n";
            }
        }
    }

    std::cout << "A wins " << a_wins << "/" << games
              << " (as black " << a_black << ", as white " << a_white << ")\n";
    std::cout << "B wins " << b_wins << "/" << games
              << " (as black " << b_black << ", as white " << b_white << ")\n";
    return 0;
}

// HTP / GTP-style protocol loop so HexGui (and twogtp-style match tools) can
// drive the engine over stdin/stdout. Coordinates use move_to_string /
// move_from_string (column letter + row number; Black connects top-bottom,
// White left-right), matching the benzene/HexGui convention.
int run_htp(const Args& args) {
    int n = get_int(args, "n", 9);
    auto rules = std::make_shared<Rules>(n);
    GameState gs(rules);
    std::vector<std::pair<int, int>> history;   // (move, color) for undo / replay
    SearchConfig cfg = config_from_args(args, n);
    auto net = load_model_if_requested(args, n);
    std::cout.setf(std::ios::unitbuf);           // flush each response (pipe-safe)

    const std::string cmds =
        "protocol_version\nname\nversion\nlist_commands\nknown_command\n"
        "boardsize\nclear_board\nplay\ngenmove\nundo\nshowboard\nfinal_score\nquit";

    auto is_known = [&](const std::string& c) {
        return (cmds + "\n").find(c + "\n") != std::string::npos;
    };
    auto parse_color = [](std::string s) -> int {
        for (auto& ch : s) ch = static_cast<char>(std::tolower((unsigned char)ch));
        if (s == "b" || s == "black") return BLACK;
        if (s == "w" || s == "white") return WHITE;
        return -1;
    };
    auto reply = [](const std::string& id, bool ok, const std::string& body) {
        std::cout << (ok ? '=' : '?') << id;
        if (!body.empty()) std::cout << ' ' << body;
        std::cout << "\n\n";
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.size() >= 3 && (unsigned char)line[0] == 0xEF &&
            (unsigned char)line[1] == 0xBB && (unsigned char)line[2] == 0xBF)
            line = line.substr(3);                       // strip UTF-8 BOM
        if (!line.empty() && line.back() == '\r') line.pop_back();   // strip CR (CRLF)
        std::size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        std::istringstream iss(line);
        std::vector<std::string> tok; std::string t;
        while (iss >> t) tok.push_back(t);
        if (tok.empty()) continue;
        std::size_t i = 0; std::string id;
        if (!tok[0].empty() && std::isdigit((unsigned char)tok[0][0])) { id = tok[0]; i = 1; }
        if (i >= tok.size()) { reply(id, false, "missing command"); continue; }
        std::string cmd = tok[i];
        for (auto& ch : cmd) ch = static_cast<char>(std::tolower((unsigned char)ch));
        std::vector<std::string> a(tok.begin() + i + 1, tok.end());

        if (cmd == "protocol_version") reply(id, true, "2");
        else if (cmd == "name") reply(id, true, "Damaten HexAI");
        else if (cmd == "version") reply(id, true, "1.0");
        else if (cmd == "list_commands") reply(id, true, cmds);
        else if (cmd == "known_command") reply(id, true, (!a.empty() && is_known(a[0])) ? "true" : "false");
        else if (cmd == "quit") { reply(id, true, ""); break; }
        else if (cmd == "boardsize") {
            int nn = n;
            if (!a.empty()) { try { nn = std::stoi(a[0]); } catch (...) { reply(id, false, "bad size"); continue; } }
            if (nn < 2 || nn > 19) { reply(id, false, "unacceptable size"); continue; }
            n = nn; rules = std::make_shared<Rules>(n); gs = GameState(rules);
            history.clear(); cfg = config_from_args(args, n); net = load_model_if_requested(args, n);
            reply(id, true, "");
        }
        else if (cmd == "clear_board") { gs = GameState(rules); history.clear(); reply(id, true, ""); }
        else if (cmd == "play") {
            if (a.size() < 2) { reply(id, false, "syntax: play <color> <move>"); continue; }
            int color = parse_color(a[0]);
            if (color < 0) { reply(id, false, "bad color"); continue; }
            std::string mv = a[1];
            for (auto& ch : mv) ch = static_cast<char>(std::tolower((unsigned char)ch));
            if (mv == "resign" || mv == "pass") { reply(id, true, ""); continue; }
            int idx = rules->move_from_string(mv);
            if (idx < 0 || gs.board[idx] != EMPTY) { reply(id, false, "illegal move"); continue; }
            gs.cur = color; gs = gs.apply(idx); history.push_back({ idx, color });
            reply(id, true, "");
        }
        else if (cmd == "genmove") {
            int color = (!a.empty() && parse_color(a[0]) >= 0) ? parse_color(a[0]) : gs.cur;
            if (gs.is_terminal() || legal_moves(gs).empty()) { reply(id, true, "resign"); continue; }
            gs.cur = color;
            SearchResult sr = search_position(gs, cfg, net.get());
            if (sr.move < 0) { reply(id, true, "resign"); continue; }
            gs = gs.apply(sr.move); history.push_back({ sr.move, color });
            reply(id, true, rules->move_to_string(sr.move));
        }
        else if (cmd == "undo") {
            if (history.empty()) { reply(id, false, "cannot undo"); continue; }
            history.pop_back();
            GameState g(rules);
            for (auto& mc : history) { g.cur = mc.second; g = g.apply(mc.first); }
            gs = g; reply(id, true, "");
        }
        else if (cmd == "showboard") {
            std::ostringstream os; os << "\n";
            for (int r = 0; r < n; ++r) {
                os << std::string(r, ' ');
                os << (r + 1 < 10 ? " " : "") << (r + 1) << ' ';
                for (int c = 0; c < n; ++c) {
                    int v = gs.board[r * n + c];
                    os << (v == BLACK ? 'B' : (v == WHITE ? 'W' : '.')) << ' ';
                }
                os << "\n";
            }
            reply(id, true, os.str());
        }
        else if (cmd == "final_score") {
            if (gs.is_terminal()) reply(id, true, gs.winner() == BLACK ? "B+" : "W+");
            else reply(id, true, "0");
        }
        else if (cmd == "hexgui-analyze_commands") reply(id, true, "");
        else reply(id, false, "unknown command");
    }
    return 0;
}

int run_legacy_args(int argc, char* argv[]) {
    int n = std::stoi(argv[1]);
    auto rules = std::make_shared<Rules>(n);
    if (argc < 2 + rules->nn + 2) {
        std::cerr << "legacy mode: hex_ai N b0 ... b(NN-1) player last\n";
        return 2;
    }
    GameState gs(rules);
    for (int i = 0; i < rules->nn; ++i) gs.board[i] = std::stoi(argv[2 + i]);
    gs.cur = std::stoi(argv[2 + rules->nn]);
    gs.last_move = std::stoi(argv[2 + rules->nn + 1]);
    rebuild_union_find(gs);
    Args args;
    SearchConfig cfg = config_from_args(args, n);
    SearchResult sr = search_position(gs, cfg, nullptr);
    std::cout << sr.move << "\n";
    return 0;
}

// Create a randomly-initialised ConvNet and save it in HEXCNN_V1 format. Used
// to validate the C++ CNN plumbing locally before any real (Colab) training.
int run_initcnn(const Args& args) {
    int n = get_int(args, "n", 9);
    int channels = get_int(args, "channels", 32);
    int blocks = get_int(args, "blocks", 4);
    std::string out = get_string(args, "out", "hex_cnn.nn");
    uint64_t seed = static_cast<uint64_t>(get_int(args, "seed", 20240601));
    ConvNet net;
    net.init(n, channels, blocks, seed);
    if (!net.save(out)) {
        std::cerr << "cnn save failed: " << out << "\n";
        return 5;
    }
    std::cout << "initialized CNN: n=" << n << " channels=" << channels
              << " blocks=" << blocks << " -> " << out << "\n";
    return 0;
}

void print_help() {
    std::cout
        << "Hex AI commands\n"
        << "  play      --n 9 --iters 2300 [--model hex_model.nn] [--human b]\n"
        << "  move      --n 9 --board <81 chars> --player 1 --last -1\n"
        << "  selfplay  --n 9 --games 20 --iters 400 --out hex_selfplay.tsv [--model hex_model.nn]\n"
        << "  train     --n 9 --data hex_selfplay.tsv --model-out hex_model.nn --epochs 5\n"
        << "  pipeline  --n 9 --cycles 3 --games 10 --iters 300 --data hex_selfplay.tsv --model hex_model.nn\n"
        << "  match     --n 9 --games 20 --model-a a.nn --model-b b.nn [--iters-a N] [--rollout-b 0]\n"
        << "  initcnn   --n 9 --channels 32 --blocks 4 --out hex_cnn.nn   (random CNN, HEXCNN_V1)\n"
        << "  eval      --n 9 --board <81 chars> --player 1 --model m.nn  (raw value+logits, no search)\n"
        << "\n"
        << "Search flags: --cpuct --fpu --rollout 0|1 --solver 0|1 --prior-blend 0..1.\n"
        << "Models: HEXNN_V1 (MLP) and HEXCNN_V1 (conv resnet) are auto-detected.\n"
        << "Board chars: . or 0 = empty, B/1 = black, W/2 = white.\n"
        << "Black connects top-bottom. White connects left-right.\n";
}

}  // namespace hexai

#ifdef __EMSCRIPTEN__
// ---------------------------------------------------------------------------
// WebAssembly entry points. Compiled only under Emscripten; the native CLI
// build is unaffected. These wrap the same internals the `move`/`eval` CLI
// commands use, so the browser engine behaves identically to HexAI.exe.
//   hex_init(n, "/hex_model.nn")  -> 1 on success (model preloaded into MEMFS)
//   hex_best_move(board, player, last, iters) -> chosen cell index (or <0)
//   hex_policy(board, player) -> "value;p0,p1,...,pNN-1" legal-masked softmax
// ---------------------------------------------------------------------------
#include <emscripten/emscripten.h>
using namespace hexai;

namespace {
    std::shared_ptr<Rules> g_wasm_rules;
    std::unique_ptr<NeuralNet> g_wasm_net;
    int g_wasm_n = 9;
    std::string g_wasm_out;  // return buffer for hex_policy
}

extern "C" {

EMSCRIPTEN_KEEPALIVE
int hex_init(int n, const char* model_path) {
    g_wasm_n = n;
    g_wasm_rules = std::make_shared<Rules>(n);
    g_wasm_net = load_any_model(model_path ? std::string(model_path) : std::string(), n);
    return (g_wasm_net && g_wasm_net->is_ready()) ? 1 : 0;
}

EMSCRIPTEN_KEEPALIVE
int hex_best_move(const char* board, int player, int last, int iters) {
    if (!g_wasm_rules) return -3;
    GameState gs(g_wasm_rules);
    if (!parse_board_text(board ? std::string(board) : std::string(), g_wasm_rules->nn, gs.board))
        return -2;
    gs.cur = player;
    gs.last_move = last;
    rebuild_union_find(gs);
    Args a;
    if (iters > 0) a.opt["iters"] = std::to_string(iters);
    SearchConfig cfg = config_from_args(a, g_wasm_n);
    SearchResult sr = search_position(gs, cfg, g_wasm_net.get());
    return sr.move;
}

EMSCRIPTEN_KEEPALIVE
const char* hex_policy(const char* board, int player) {
    g_wasm_out.clear();
    if (!g_wasm_net || !g_wasm_net->is_ready()) { g_wasm_out = "error;no-model"; return g_wasm_out.c_str(); }
    GameState gs(g_wasm_rules);
    if (!parse_board_text(board ? std::string(board) : std::string(), g_wasm_rules->nn, gs.board)) {
        g_wasm_out = "error;bad-board"; return g_wasm_out.c_str();
    }
    const int nn = g_wasm_rules->nn;
    Eval e = g_wasm_net->evaluate(gs.board, player);
    // Legal-masked softmax over empty cells (same as the Python /policy bridge).
    double mx = -1e300;
    for (int i = 0; i < nn; ++i) if (gs.board[i] == 0 && e.logits[i] > mx) mx = e.logits[i];
    std::vector<double> p(nn, 0.0);
    double z = 0.0;
    for (int i = 0; i < nn; ++i) if (gs.board[i] == 0) { p[i] = std::exp(e.logits[i] - mx); z += p[i]; }
    if (z <= 0.0) z = 1.0;
    std::ostringstream os;
    os << std::setprecision(6) << e.value;
    for (int i = 0; i < nn; ++i) os << (i == 0 ? ';' : ',') << (p[i] / z);
    g_wasm_out = os.str();
    return g_wasm_out.c_str();
}

}  // extern "C"
#endif  // __EMSCRIPTEN__

#ifndef __EMSCRIPTEN__
int main(int argc, char* argv[]) {
    using namespace hexai;
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc > 1 && is_integer_text(argv[1])) return run_legacy_args(argc, argv);

    Args args = parse_args(argc, argv);
    if (args.command == "help" || args.command == "--help" || args.command == "-h") {
        print_help();
        return 0;
    }
    if (args.command == "play") return run_play(args);
    if (args.command == "move") return run_move(args);
    if (args.command == "selfplay") return run_selfplay(args);
    if (args.command == "train") return run_train(args);
    if (args.command == "pipeline") return run_pipeline(args);
    if (args.command == "match") return run_match(args);
    if (args.command == "initcnn") return run_initcnn(args);
    if (args.command == "eval") return run_eval(args);
    if (args.command == "htp" || args.command == "gtp") return run_htp(args);

    std::cerr << "unknown command: " << args.command << "\n";
    print_help();
    return 1;
}
#endif  // __EMSCRIPTEN__
