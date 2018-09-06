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
    ImpFloat eta, lambda;
    ImpInt nr_pass, k, nr_threads;
    string model_path, predict_path;
    bool item_bias;
    Parameter():eta(0.1), lambda(1e-5), nr_pass(20), k(4), nr_threads(1), item_bias(true) {};
};

class Node {
public:
    ImpLong idx;
    ImpDouble val;
    Node(): idx(0), val(0) {};
};

class ImpData {
public:
    string file_name;
    ImpLong l, n, m;
    vector<Node> M, N;
    vector<Node*> X, Y;

    ImpData(string file_name): file_name(file_name), l(0), n(0), m(0) {};
    void read(bool has_label, ImpLong max_m=ULONG_MAX);
    void print_data_info();
    void transY(const vector<Node*> &YT);
};


class ImpProblem {
public:
    ImpProblem(shared_ptr<ImpData> &Tr, shared_ptr<ImpData> &Te,
            shared_ptr<ImpData> &Xt, shared_ptr<Parameter> &param)
        :Tr(Tr), Te(Te), Xt(Xt), param(param) {};

    void init();
    void solve();

private:
    ImpDouble loss, reg, lambda, w;
    shared_ptr<ImpData> Tr, Te, Xt;
    shared_ptr<Parameter> param;

    ImpInt k;
    ImpLong l, n, mc, mt;

    Vec U, V, gu, gv;
    Vec P, Q, CTC;

    Vec va_loss;
    vector<ImpInt> top_k, orders;

    void UTx(Node *x0, Node* x1, Vec &A, ImpDouble *c);
    void UTX(shared_ptr<ImpData> &D, Vec &A, Vec &C);

    void QTQ(const Vec &C, const ImpLong &l);

    void one_epoch();

    void gd(shared_ptr<ImpData> &data, Vec &A, Vec &C, Vec &D, Vec &G);
    void cg(shared_ptr<ImpData> &data, Vec &A, Vec &D, Vec &G);
    void Hs(shared_ptr<ImpData> &data, Vec &A, Vec &D, Vec &S, Vec &HS);
    void func();



    void init_va_loss(ImpInt size);

    void pred_z(ImpLong i, Vec &z);
    void pred_items();
    void prec_k(Vec &z, ImpLong i, vector<ImpInt> &top_k, vector<ImpLong> &hit_counts);
    void validate();
    void print_epoch_info(ImpInt t);

};
