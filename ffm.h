#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <memory>
#include <cstring>
#include <stdlib.h>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <climits>
#include <utility>
#include <numeric>
#include <cassert>


#include <immintrin.h>

#include<omp.h>
#include "mkl.h"



using namespace std;

typedef double ImpFloat;
typedef double ImpDouble;
typedef unsigned int ImpInt;
typedef unsigned long int ImpLong;
typedef vector<ImpDouble> Vec;

const ImpInt MIN_Z = -10000;

class Parameter {
public:
    ImpFloat omega, lambda;
    ImpInt nr_pass, k, nr_threads;
    string model_path, predict_path;
    bool item_bias;
    Parameter():omega(0.1), lambda(1e-5), nr_pass(20), k(4), nr_threads(1), item_bias(true) {};
};

class Node {
public:
    ImpInt fid;
    ImpLong idx;
    ImpDouble val;
    Node(): fid(0), idx(0), val(0) {};
};

class ImpData {
public:
    string file_name;
    ImpLong m, n, f, nnz_x, nnz_y;
    vector<Node> M, N;
    vector<Node*> X, Y;


    vector<vector<Node>> Ns;
    vector<vector<Node*>> Xs;
    vector<ImpLong> Ds;

    ImpData(string file_name): file_name(file_name), m(0), n(0), f(0) {};
    void read(bool has_label, ImpLong max_m=ULONG_MAX);
    void print_data_info();
    void split_fields();
    void transY(const vector<Node*> &YT);
};


class ImpProblem {
public:
    ImpProblem(shared_ptr<ImpData> &U, shared_ptr<ImpData> &Ut,
            shared_ptr<ImpData> &V, shared_ptr<Parameter> &param)
        :U(U), Ut(Ut), V(V), param(param) {};

    void init();
    void solve();

private:
    ImpDouble loss, reg, lambda, w;
    shared_ptr<ImpData> U, Ut, V;
    shared_ptr<Parameter> param;

    ImpInt k, fu, fv, f;
    ImpLong m, n;
    ImpLong mt;

    vector<Vec> W, H, P, Q, Pt, Qt;
    Vec a, b, va_loss, CTC;

    vector<ImpInt> top_k;

    void init_pair(const ImpInt &f12, const ImpInt &fi, const ImpInt &fj,
            const shared_ptr<ImpData> &d1, const shared_ptr<ImpData> &d2);

    void add_side(const Vec &p, const Vec &q, const ImpLong &m1, Vec &a1);
    void calc_side();
    void init_y_tilde();
    ImpDouble calc_cross(const ImpLong &i, const ImpLong &j);

    void update_side(const ImpInt &f1, const ImpInt &f2, bool add);
    void update_cross(const ImpInt &f1, const ImpInt &f2, bool add);

    void UTx(Node *x0, Node* x1, Vec &A, ImpDouble *c);
    void UTX(const vector<Node*> &X, ImpLong m1, Vec &A, Vec &C);


    void QTQ(const Vec &C, const ImpLong &l);

    void solve_side(const ImpInt &f1, const ImpInt &f2);
    void solve_cross(const ImpInt &f1, const ImpInt &f2);


    void one_epoch();
    void init_va(ImpInt size);

    void pred_z(ImpLong i, Vec &z);
    void pred_items();
    void prec_k(Vec &z, ImpLong i, vector<ImpInt> &top_k, vector<ImpLong> &hit_counts);
    void validate();
    void print_epoch_info(ImpInt t);

};
