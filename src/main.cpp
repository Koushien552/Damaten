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
        UnionFind copy = uf;
        if (player == BLACK) return copy.connected(rules->b_top, rules->b_bottom);
        return copy.connected(rules->w_left, rules->w_right);
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
    std::vector<int> before = board;
    before[last_opp] = EMPTY;

    std::vector<int> stones;
    stones.reserve(rules.nn);
    for (int i = 0; i < rules.nn; ++i) {
        if (before[i] == player) stones.push_back(i);
    }

    std::vector<int> mark(rules.nn, 0);
    for (std::size_t ia = 0; ia < stones.size(); ++ia) {
        int a = stones[ia];
        std::set<int> empty_a;
        for (int k = 0; k < rules.nbr_count[a]; ++k) {
            int nb = rules.nbrs[a][k];
            if (before[nb] == EMPTY) empty_a.insert(nb);
        }
        for (std::size_t ib = ia + 1; ib < stones.size(); ++ib) {
            int b = stones[ib];
            bool adjacent = false;
            for (int k = 0; k < rules.nbr_count[a]; ++k) {
                if (rules.nbrs[a][k] == b) {
                    adjacent = true;
                    break;
                }
            }
            if (adjacent) continue;

            std::vector<int> common;
            for (int k = 0; k < rules.nbr_count[b]; ++k) {
                int nb = rules.nbrs[b][k];
                if (before[nb] == EMPTY && empty_a.count(nb)) common.push_back(nb);
            }
            if (common.size() < 2) continue;
            bool threatened = std::find(common.begin(), common.end(), last_opp) != common.end();
            if (!threatened) continue;
            for (int x : common) {
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

struct TinyNet {
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

std::vector<double> network_or_heuristic_priors(const GameState& gs, const TinyNet* net) {
    if (!net || !net->ready || net->n != gs.rules->n) return heuristic_priors(gs);
    NetForward f = net->infer(gs.board, gs.cur);
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
    if (sum <= 0.0 || !std::isfinite(sum)) return heuristic_priors(gs);
    for (double& p : priors) p /= sum;
    return priors;
}

struct SearchConfig {
    int iterations = 1500;
    int endgame_depth = 5;
    double exploration = 1.15;
    double rollout_weight = 0.45;
    bool use_rollout = true;
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

    MCTSNode(const GameState& state, MCTSNode* par, int mv, double prior, const TinyNet* net)
        : st(state), parent(par), priors(network_or_heuristic_priors(state, net)), move(mv), prior_from_parent(prior) {
        untried = legal_moves(st);
    }

    double child_score(const MCTSNode& child, double exploration) const {
        double q = 0.5;
        if (child.visits > 0) {
            double black_q = child.black_wins / child.visits;
            q = st.cur == BLACK ? black_q : (1.0 - black_q);
        }
        double u = exploration * child.prior_from_parent * std::sqrt(static_cast<double>(visits + 1)) / (1.0 + child.visits);
        return q + u;
    }

    MCTSNode* best_child(double exploration) {
        return std::max_element(children.begin(), children.end(), [&](const auto& a, const auto& b) {
            return child_score(*a, exploration) < child_score(*b, exploration);
        })->get();
    }

    int pop_best_untried() {
        int best_pos = -1;
        double best_prior = -1.0;
        for (int i = 0; i < static_cast<int>(untried.size()); ++i) {
            int m = untried[i];
            if (priors[m] > best_prior) {
                best_prior = priors[m];
                best_pos = i;
            }
        }
        if (best_pos < 0) return -1;
        int move_out = untried[best_pos];
        untried[best_pos] = untried.back();
        untried.pop_back();
        return move_out;
    }
};

double evaluate_leaf_black_win(const GameState& st, const TinyNet* net, const SearchConfig& config) {
    if (st.has_won(BLACK)) return 1.0;
    if (st.has_won(WHITE)) return 0.0;

    double heuristic = heuristic_black_win(st);
    double value = heuristic;
    if (config.use_rollout) {
        double rollout = rollout_light(st);
        value = (1.0 - config.rollout_weight) * heuristic + config.rollout_weight * rollout;
    }
    if (net && net->ready && net->n == st.rules->n) {
        NetForward f = net->infer(st.board, st.cur);
        double current_win_prob = (f.value + 1.0) * 0.5;
        double net_black = st.cur == BLACK ? current_win_prob : (1.0 - current_win_prob);
        value = 0.50 * net_black + 0.30 * value + 0.20 * heuristic;
    }
    return std::max(0.0, std::min(1.0, value));
}

SearchResult mcts_search(const GameState& root_state, const SearchConfig& config, const TinyNet* net) {
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

    for (int iter = 0; iter < config.iterations; ++iter) {
        MCTSNode* node = &root;
        std::vector<MCTSNode*> path;
        path.push_back(node);

        while (node->untried.empty() && !node->children.empty() && !node->st.is_terminal()) {
            node = node->best_child(config.exploration);
            path.push_back(node);
        }

        if (!node->st.is_terminal() && !node->untried.empty()) {
            int m = node->pop_best_untried();
            GameState ns = node->st.apply(m);
            double prior = node->priors[m] > 0.0 ? node->priors[m] : 1.0 / std::max(1, node->st.empty_count());
            node->children.push_back(std::make_unique<MCTSNode>(ns, node, m, prior, net));
            node = node->children.back().get();
            path.push_back(node);
        }

        double black_win = evaluate_leaf_black_win(node->st, net, config);
        for (MCTSNode* p : path) {
            ++p->visits;
            p->black_wins += black_win;
        }
    }

    MCTSNode* best = nullptr;
    for (const auto& child : root.children) {
        result.visits[child->move] = child->visits;
        if (!best || child->visits > best->visits) best = child.get();
    }
    if (best) {
        result.move = best->move;
        result.black_win_estimate = best->visits > 0 ? best->black_wins / best->visits : 0.5;
    } else {
        auto moves = legal_moves(root_state);
        result.move = moves.empty() ? -1 : moves.front();
    }
    return result;
}

SearchResult search_position(const GameState& gs, const SearchConfig& config, const TinyNet* net) {
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
    cfg.iterations = get_int(args, "iters", n <= 7 ? 2500 : (n <= 9 ? 1500 : 700));
    cfg.endgame_depth = get_int(args, "endgame", n <= 7 ? 8 : (n <= 9 ? 5 : 3));
    cfg.exploration = get_double(args, "cpuct", 1.15);
    cfg.rollout_weight = get_double(args, "rollout-weight", 0.45);
    cfg.use_rollout = get_int(args, "rollout", 1) != 0;
    return cfg;
}

std::unique_ptr<TinyNet> load_model_if_requested(const Args& args, int n) {
    std::string model_path = get_string(args, "model", "");
    if (model_path.empty()) return nullptr;
    auto net = std::make_unique<TinyNet>();
    if (!net->load(model_path)) {
        std::cerr << "model load failed: " << model_path << "\n";
        return nullptr;
    }
    if (net->n != n) {
        std::cerr << "model board size mismatch. model=" << net->n << " requested=" << n << "\n";
        return nullptr;
    }
    return net;
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

int run_selfplay(const Args& args) {
    int n = get_int(args, "n", 9);
    int games = get_int(args, "games", 10);
    int temp_moves = get_int(args, "temp-moves", n <= 7 ? 10 : 18);
    std::string out_path = get_string(args, "out", "hex_selfplay.tsv");
    g_rng = Random(static_cast<uint64_t>(get_int(args, "seed", 123456789)));
    auto rules = std::make_shared<Rules>(n);
    SearchConfig cfg = config_from_args(args, n);
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

void print_help() {
    std::cout
        << "Hex AI commands\n"
        << "  play      --n 9 --iters 1500 [--model hex_model.nn] [--human b]\n"
        << "  move      --n 9 --board <81 chars> --player 1 --last -1\n"
        << "  selfplay  --n 9 --games 20 --iters 400 --out hex_selfplay.tsv [--model hex_model.nn]\n"
        << "  train     --n 9 --data hex_selfplay.tsv --model-out hex_model.nn --epochs 5\n"
        << "  pipeline  --n 9 --cycles 3 --games 10 --iters 300 --data hex_selfplay.tsv --model hex_model.nn\n"
        << "\n"
        << "Board chars: . or 0 = empty, B/1 = black, W/2 = white.\n"
        << "Black connects top-bottom. White connects left-right.\n";
}

}  // namespace hexai

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

    std::cerr << "unknown command: " << args.command << "\n";
    print_help();
    return 1;
}
