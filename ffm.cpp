#include "ffm.h"

ImpDouble qrsqrt(ImpDouble x)
{
    ImpDouble xhalf = 0.5*x;
    ImpLong i;
    memcpy(&i, &x, sizeof(i));
    i = 0x5fe6eb50c7b537a9 - (i>>1);
    memcpy(&x, &i, sizeof(i));
    x = x*(1.5 - xhalf*x*x);
    return x;
}

ImpDouble sum(const Vec &v) {
    ImpDouble sum = 0;
    for (ImpDouble val: v)
        sum += val;
    return sum;
}

void axpy(const ImpDouble *x, ImpDouble *y, const ImpLong &l, const ImpDouble &lambda) {
    cblas_daxpy(l, lambda, x, 1, y, 1);
}

void mm(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong l, const ImpLong n, const ImpInt k) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            l, n, k, 1, a, k, b, n, 0, c, n);
}

void mm(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong l, const ImpLong n, const ImpInt k, const ImpDouble &beta) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            l, n, k, 1, a, k, b, n, beta, c, n);
}

void mm(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong k, const ImpInt l) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            k, k, l, 1, a, k, b, k, 0, c, k);
}

void mv(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong l, const ImpInt k, const ImpDouble &beta, bool trans) {
    const CBLAS_TRANSPOSE CBTr= (trans)? CblasTrans: CblasNoTrans;
    cblas_dgemv(CblasRowMajor, CBTr, l, k, 1, a, k, b, 1, beta, c, 1);
}

const ImpInt index_vec(const ImpInt f1, const ImpInt f2, const ImpInt f) {
    return f2 + (f-1)*f1 - f1*(f1-1)/2;
}

ImpDouble inner(const ImpDouble *p, const ImpDouble *q, const ImpInt k)
{
    __m128d XMM = _mm_setzero_pd();
    for(ImpInt d = 0; d < k; d += 2)
        XMM = _mm_add_pd(XMM, _mm_mul_pd(
                  _mm_load_pd(p+d), _mm_load_pd(q+d)));
    XMM = _mm_hadd_pd(XMM, XMM);
    ImpDouble product;
    _mm_store_sd(&product, XMM);
    return product;
}

void init_mat(Vec &vec, const ImpLong nr_rows, const ImpLong nr_cols) {
    default_random_engine ENGINE(rand());
    vec.resize(nr_rows*nr_cols);
    uniform_real_distribution<ImpDouble> dist(0, 0.1*qrsqrt(nr_cols));

    auto gen = std::bind(dist, ENGINE);
    generate(vec.begin(), vec.end(), gen);
}

void ImpData::read(bool has_label, ImpLong max_m) {
    ifstream fs(file_name);
    string line, label_block, label_str;
    char dummy;

    ImpLong fid, idx, y_nnz=0, x_nnz=0;
    ImpDouble val;

    while (getline(fs, line)) {
        m++;
        istringstream iss(line);

        if (has_label) {
            iss >> label_block;
            istringstream labelst(label_block);
            while (getline(labelst, label_str, ',')) {
                idx = stoi(label_str);
                n = max(n, idx+1);
                y_nnz++;
            }
        }

        while (iss >> fid >> dummy >> idx >> dummy >> val) {
            f = max(f, fid+1);
            x_nnz++;
        }
    }

    fs.clear();
    fs.seekg(0);

    nnz_x = x_nnz;
    N.resize(x_nnz);

    X.resize(m+1);
    Y.resize(m+1);

    if (has_label) {
        nnz_y = nnz_y;
        M.resize(y_nnz);
    }

    vector<ImpInt> nnx(m, 0), nny(m, 0);

    ImpLong nnz_i=0, nnz_j=0, i=0;

    while (getline(fs, line)) {
        istringstream iss(line);

        if (has_label) {
            iss >> label_block;
            istringstream labelst(label_block);
            while (getline(labelst, label_str, ',')) {
                nnz_j++;
                ImpLong idx = stoi(label_str);
                M[nnz_j-1].idx = idx;
            }
            nny[i] = nnz_j;
        }

        while (iss >> fid >> dummy >> idx >> dummy >> val) {
            nnz_i++;
            N[nnz_i-1].fid = fid;
            N[nnz_i-1].idx = idx;
            N[nnz_i-1].val = val;
        }
        nnx[i] = nnz_i;
        i++;
    }

    X[0] = N.data();
    for (ImpLong i = 0; i < m; i++) {
        X[i+1] = N.data() + nnx[i];
    }

    if (has_label) {
        Y[0] = M.data();
        for (ImpLong i = 0; i < m; i++) {
            Y[i+1] = M.data() + nny[i];
        }
    }
    fs.close();
}

void ImpData::split_fields() {
    Ns.resize(f);
    Xs.resize(f);
    Ds.resize(f);

    vector<ImpLong> f_sum_nnz(f, 0);
    vector<vector<ImpLong>> f_nnz(f);

    for (ImpInt fi = 0; fi < f; fi++) {
        Ds[fi] = 0;
        f_nnz[fi].resize(m, 0);
        Xs[fi].resize(m+1);
    }

    for (ImpLong i = 0; i < m; i++) {
        for (Node* x = X[i]; x < X[i+1]; x++) {
            ImpInt fid = x->fid;
            f_sum_nnz[fid]++;
            f_nnz[fid][i]++;
        }
    }

    for (ImpInt fi = 0; fi < f; fi++) {
        Ns[fi].resize(f_sum_nnz[fi]);
        f_sum_nnz[fi] = 0;
    }

    for (ImpLong i = 0; i < m; i++) {
        for (Node* x = X[i]; x < X[i+1]; x++) {
            ImpInt fid = x->fid;
            ImpLong idx = x->idx;
            ImpDouble val = x->val;

            f_sum_nnz[fid]++;
            Ds[fid] = max(idx+1, Ds[fid]);
            ImpLong nnz_i = f_sum_nnz[fid]-1;

            Ns[fid][nnz_i].fid = fid;
            Ns[fid][nnz_i].idx = idx;
            Ns[fid][nnz_i].val = val;
        }
    }

    for (ImpInt fi = 0; fi < f; fi++) {
        Node* fM = Ns[fi].data();
        Xs[fi][0] = fM;
        for (ImpLong i = 0; i < m; i++)
            Xs[fi][i+1] = fM + f_nnz[fi][i];
    }

    X.clear();
    X.shrink_to_fit();

    N.clear();
    N.shrink_to_fit();
}

void ImpData::transY(const vector<Node*> &YT) {
    n = YT.size() - 1;
    vector<pair<ImpLong, Node*>> perm;
    ImpLong nnz = 0;
    vector<ImpLong> nnzs(m, 0);

    for (ImpLong i = 0; i < n; i++)
        for (Node* y = YT[i]; y < YT[i+1]; y++) {
            if (y->idx >= m )
              continue;
            nnzs[y->idx]++;
            perm.emplace_back(i, y);
            nnz++;
        }

    auto sort_by_column = [&] (const pair<ImpLong, Node*> &lhs,
            const pair<ImpLong, Node*> &rhs) {
        return tie(lhs.second->idx, lhs.first) < tie(rhs.second->idx, rhs.first);
    };

    sort(perm.begin(), perm.end(), sort_by_column);

    M.resize(nnz);
    nnz_y = nnz;
    for (ImpLong nnz_i = 0; nnz_i < nnz; nnz_i++) {
        M[nnz_i].idx = perm[nnz_i].first;
        //M[nnz_i].val = perm[nnz_i].second->val;
    }

    Y[0] = M.data();
    ImpLong start_idx = 0;
    for (ImpLong i = 0; i < m; i++) {
        start_idx += nnzs[i];
        Y[i+1] = M.data()+start_idx;
    }
}

void ImpData::print_data_info() {
    cout << "File:";
    cout << file_name;
    cout.width(12);
    cout << "m:";
    cout << m;
    cout.width(12);
    cout << "n:";
    cout << n;
    cout.width(12);
    cout << "f:";
    cout << f;
    cout.width(12);
    cout << endl;
}

void ImpProblem::UTx(const Node* x0, const Node* x1, const Vec &A, ImpDouble *c) {
    for (const Node* x = x0; x < x1; x++) {
        const ImpLong idx = x->idx;
        const ImpDouble val = x->val;
        for (ImpInt d = 0; d < k; d++) {
            ImpLong jd = idx*k+d;
            c[d] += val*A[jd];
        }
    }
}

void ImpProblem::UTX(const vector<Node*> &X, const ImpLong m1, const Vec &A, Vec &C) {
    fill(C.begin(), C.end(), 0);

    ImpDouble* c = C.data();
    for (ImpLong i = 0; i < m1; i++)
        UTx(X[i], X[i+1], A, c+i*k);
}


void ImpProblem::init_pair(const ImpInt &f12,
        const ImpInt &fi, const ImpInt &fj,
        const shared_ptr<ImpData> &d1,
        const shared_ptr<ImpData> &d2) {
    const ImpLong Df1 = d1->Ds[fi];
    const ImpLong Df2 = d2->Ds[fj];

    const vector<Node*> &X1 = d1->Xs[fi];
    const vector<Node*> &X2 = d2->Xs[fj];

    init_mat(W[f12], Df1, k);
    init_mat(H[f12], Df2, k);
    P[f12].resize(d1->m*k, 0);
    Q[f12].resize(d2->m*k, 0);
    UTX(X1, d1->m, W[f12], P[f12]);
    UTX(X2, d2->m, H[f12], Q[f12]);
}

void ImpProblem::add_side(const Vec &p, const Vec &q, const ImpLong &m1, Vec &a1) {
    const ImpDouble *pp = p.data(), *qp = q.data();
    for (ImpLong i = 0; i < m1; i++) {
        const ImpDouble *pi = pp+i*k, *qi = qp+i*k;
        a1[i] += inner(pi, qi, k);
    }
}

void ImpProblem::calc_side() {
    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = f1; f2 < fu; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            add_side(P[f12], Q[f12], m, a);
        }
    }
    for (ImpInt f1 = 0; f1 < fv; f1++) {
        for (ImpInt f2 = f1; f2 < fv; f2++) {
            const ImpInt f12 = index_vec(fu+f1, fu+f2, f);
            add_side(P[f12], Q[f12], n, b);
        }
    }
}

ImpDouble ImpProblem::calc_cross(const ImpLong &i, const ImpLong &j) {
    ImpDouble cross_value = 0.0;
    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = 0; f2 < fv; f2++) {
            const ImpInt f12 = index_vec(f1, fu+f2, f);
            const ImpDouble *pp = P[f12].data();
            const ImpDouble *qp = Q[f12].data();
            cross_value += inner(pp+i*k, qp+j*k, k);
        }
    }
    return cross_value;
}

void ImpProblem::init_y_tilde() {
    for (ImpLong i = 0; i < m; i++) {
        for (Node* y = U->Y[i]; y < U->Y[i+1]; y++) {
            ImpLong j = y->idx;
            y->val = 1-a[i]-b[j]-calc_cross(i, j);
        }
    }
    for (ImpLong j = 0; j < n; j++) {
        for (Node* y = V->Y[j]; y < V->Y[j+1]; y++) {
            ImpLong i = y->idx;
            y->val = 1-a[i]-b[j]-calc_cross(i, j);
        }
    }
}

void ImpProblem::update_side(const ImpInt &f1, const ImpInt &f2, bool add) {
    const ImpLong f12 = index_vec(f1, f2, f);
    const ImpDouble *pp = P[f12].data(), *qp = Q[f12].data();

    Vec &a1 = (f1 < fu)? a:b;
    shared_ptr<ImpData> U1 = (f1 < fu)? U:V;
    shared_ptr<ImpData> V1 = (f1 < fu)? V:U;

    const int flag = add*2-1;
    Vec gaps(U1->m);

    for (ImpLong i = 0; i < U1->m; i++) {
        gaps[i] = inner(pp+i*k, qp+i*k, k);
        a1[i] += flag*gaps[i];
        for (Node* y = U1->Y[i]; y < U1->Y[i+1]; y++) {
            y->val += flag*gaps[i];
        }
    }
    for (ImpLong j = 0; j < V1->m; j++) {
        for (Node* y = V1->Y[j]; y < V1->Y[j+1]; y++) {
            const ImpLong i = y->idx;
            y->val += flag*gaps[i];
        }
    }
}

void ImpProblem::update_cross(const ImpInt &f1, const ImpInt &f2, bool add) {
    const ImpLong f12 = index_vec(f1, f2, f);
    const ImpDouble *pp = P[f12].data(), *qp = Q[f12].data();
    const int flag = add*2-1;
    for (ImpLong i = 0; i < m; i++) {
        for (Node* y = U->Y[i]; y < U->Y[i+1]; y++) {
            ImpLong j = y->idx;
            y->val += flag*inner(pp+i*k, qp+j*k, k);
        }
    }
    for (ImpLong j = 0; j < n; j++) {
        for (Node* y = V->Y[j]; y < V->Y[j+1]; y++) {
            ImpLong i = y->idx;
            y->val += flag*inner(pp+i*k, qp+j*k, k);
        }
    }
}


void ImpProblem::init() {
    lambda = param->lambda;
    w = param->omega;
    r = param->r;

    m = U->m;
    n = V->m;

    fu = U->f;
    fv = V->f;
    f = fu+fv;

    k = param->k;

    a.resize(m, 0);
    b.resize(n, 0);

    sa.resize(m, 0);
    sb.resize(n, 0);

    CTC.resize(k*k);

    const ImpInt nr_blocks = f*(f+1)/2;

    W.resize(nr_blocks);
    H.resize(nr_blocks);

    P.resize(nr_blocks);
    Q.resize(nr_blocks);

    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? U: V);
        const ImpInt fi = ((f1>=fu)? f1-fu: f1);
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? U: V);
            const ImpInt fj = ((f2>=fu)? f2-fu: f2);
            const ImpInt f12 = index_vec(f1, f2, f);
            init_pair(f12, fi, fj, d1, d2);
        }
    }

    cache_sasb();
    calc_side();
    init_y_tilde();
}

void ImpProblem::cache_sasb() {
    fill(sa.begin(), sa.end(), 0);
    fill(sb.begin(), sb.end(), 0);

    const Vec o1(m, 1), o2(n, 1);
    Vec tk(k);

    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = fu; f2 < f; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            const Vec &P1 = P[f12], &Q1 = Q[f12];

            fill(tk.begin(), tk.end(), 0);
            mv(Q1.data(), o2.data(), tk.data(), n, k, 0, true);
            mv(P1.data(), tk.data(), sa.data(), m, k, 1, false);

            fill(tk.begin(), tk.end(), 0);
            mv(P1.data(), o1.data(), tk.data(), m, k, 0, true);
            mv(Q1.data(), tk.data(), sb.data(), n, k, 1, false);
        }
    }
}

void ImpProblem::gd_side(const ImpInt &f1, const Vec &Q1, Vec &G) {

    const ImpInt base = (f1 < fu)? 0: fu;
    const ImpInt fi = f1-base;

    const ImpDouble *qp = Q1.data();

    const shared_ptr<ImpData> U1 = (f1 < fu)? U:V;
    const vector<Node*> &Y = U1->Y;
    const vector<Node*> &X = U1->Xs[fi];

    const ImpLong m1 = (f1 < fu)? m:n;
    const ImpLong n1 = (f1 < fu)? n:m;

    const Vec &b1 = (f1 < fu)? b:a;
    const Vec &a1 = (f1 < fu)? a:b;
    const Vec &sa1 = (f1 < fu)? sa:sb;
    const ImpDouble b_sum = sum(b1);


    for (ImpLong i = 0; i < m1; i++) {
        const ImpDouble *q1 = qp+i*k; 
        ImpDouble z_i = w*(n1*(r-a1[i])-b_sum-sa1[i]);

        for (Node* y = Y[i]; y < Y[i+1]; y++) {
            const ImpDouble y_tilde = y->val;
            z_i += y_tilde*(1-w)-w*(r-1);
        }

        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++)
                G[idx*k+d] += q1[d]*val*z_i;
        }
    }
}

void ImpProblem::hs_side(const ImpLong &m1, const ImpLong &n1,
        const Vec &S, Vec &Hs, const Vec &Q1, const vector<Node*> &UX, const vector<Node*> &Y) {
    const ImpDouble *qp = Q1.data();

    axpy(S.data(), Hs.data(), S.size(), lambda);

    for (ImpLong i = 0; i < m1; i++) {
        const ImpDouble* q1 = qp+i*k; 
        ImpDouble d_1 = 0;
        for (Node* y = Y[i]; y < Y[i+1]; y++)
            d_1++;
        d_1 = (1-w)*d_1 + w*n1;

        ImpDouble z_1 = 0;
        for (Node* x = UX[i]; x < UX[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++)
                z_1 += q1[d]*val*S[idx*k+d]*d_1;
        }
        for (Node* x = UX[i]; x < UX[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++)
                Hs[idx*k+d] += q1[d]*val*z_1;
        }
    }
}

void ImpProblem::gd_cross(const ImpInt &f1, const ImpInt &f12, const Vec &Q1, Vec &G) {

    const Vec &a1 = (f1 < fu)? a: b;
    const Vec &b1 = (f1 < fu)? b: a;

    const vector<Vec> &Ps = (f1 < fu)? P:Q;
    const vector<Vec> &Qs = (f1 < fu)? Q:P;

    const ImpLong &m1 = (f1 < fu)? m:n;
    const ImpLong &n1 = (f1 < fu)? n:m;

    const vector<Node*> &X = (f1 < fu)? U->Xs[f1]:V->Xs[f1-fu];
    const vector<Node*> &Y = (f1 < fu)? U->Y: V->Y;

    Vec QTQ(k*k, 0), T(m1*k), o1(n1, 1), oQ(k, 0), bQ(k, 0);

    mv(Q1.data(), o1.data(), oQ.data(), k, n, 0, true);
    mv(Q1.data(), b1.data(), bQ.data(), k, n, 0, true);

    for (ImpInt al = 0; al < fu; al++) {
        for (ImpInt be = fu; be < f; be++) {
            const ImpInt fab = index_vec(al, be, f);
            if (fab == f12) continue;
            const Vec &Qa = Qs[f12], &Pa = Ps[f12];
            mm(Qa.data(), Q1.data(), QTQ.data(), k, n1);
            mm(Pa.data(), QTQ.data(), T.data(), m1, k, k, 1);
        }
    }

    const ImpDouble *tp = T.data(), *qp = Q1.data();

    for (ImpLong i = 0; i < m1; i++) {
        Vec pi(k, 0);
        const ImpDouble *t1 = tp+i*k;
        for (Node* y = Y[i]; y < Y[i+1]; y++) {
            const ImpDouble scale = (1-w)*y->val-w*(r-1);
            const ImpLong j = y->val;
            const ImpDouble *q1 = qp+j*k;
            for (ImpInt d = 0; d < k; d++)
                pi[d] += scale*q1[d];
        }

        const ImpDouble z_i = n1*(r-a1[i]);
        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++)
                G[idx*k+d] += (pi[d]+w*(t1[d]+z_i*oQ[d]+bQ[d]))*val;
        }
    }
}


void ImpProblem::hs_cross(const ImpLong &m1, const ImpLong &n1,
        const Vec &S, Vec &Hs, const Vec &Q1, const vector<Node*> &X, const vector<Node*> &Y) {
    
    const ImpDouble *qp = Q1.data();
    axpy(S.data(), Hs.data(), S.size(), lambda);

    Vec QTQ(k*k, 0), T(m1*k, 0);

    mm(Q1.data(), Q1.data(), QTQ.data(), k, n1);
    mm(S.data(), QTQ.data(), T.data(), m1, k, k);

    for (ImpLong i = 0; i < m1; i++) {

        Vec tau(k, 0), phi(k, 0), ka(k, 0);
        UTx(X[i], X[i+1], S, phi.data());
        UTx(X[i], X[i+1], T, tau.data());

        for (Node* y = Y[i]; y < Y[i+1]; y++) {
            const ImpLong idx = y->idx;
            const ImpDouble *dp = qp + idx*k;
            const ImpDouble val = inner(phi.data(), dp, k);
            for (ImpInt d = 0; d < k; d++)
                ka[d] += val*dp[d];
        }

        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++) {
                const ImpLong jd = idx*k+d;
                Hs[jd] += ((1-w)*ka[d]+w*tau[d])*val;
            }
        }
    }
}

void ImpProblem::cg(const ImpInt &f1, const ImpInt &f2, Vec &W1,
        const Vec &Q1, const Vec &G, Vec &P1) {

    const ImpInt base = (f1 < fu)? 0: fu;
    const ImpInt fi = f1-base;

    const shared_ptr<ImpData> U1 = (f1 < fu)? U:V;
    const vector<Node*> &Y = U1->Y;
    const vector<Node*> &X = U1->Xs[fi];

    const ImpLong m1 = (f1 < fu)? m:n;
    const ImpLong n1 = (f1 < fu)? n:m;

    const ImpLong Df1 = U1->Ds[fi], Df1k = Df1*k;

    ImpInt nr_cg = 0, max_cg = 25;
    ImpDouble g2 = 0, r2, cg_eps = 1e-5, alpha = 0, beta = 0, gamma = 0, sHs;

    Vec S(Df1k, 0), R(Df1k, 0), Hs(Df1k, 0);

    for (ImpLong jd = 0; jd < Df1k; jd++) {
        R[jd] = -G[jd];
        S[jd] = R[jd];
        g2 += G[jd]*G[jd];
    }

    r2 = g2;

    while (g2*cg_eps < r2 && nr_cg < max_cg) {
        nr_cg++;

        for (ImpLong jd = 0; jd < Df1k; jd++)
            S[jd] = R[jd] + beta*S[jd];

        if ((f1 < fu && f2 < fu) || (f1>fu+1 && f2>fu+1))
            hs_side(m1, n1, S, Hs, Q1, X, Y);
        else
            hs_cross(m1, n1, S, Hs, Q1, X, Y);

        sHs = inner(S.data(), Hs.data(), Df1k);
        gamma = r2;
        alpha = gamma/sHs;
        for (ImpLong jd = 0; jd < Df1k; jd++) {
            W1[jd] += alpha*S[jd];
            R[jd] -= alpha*Hs[jd];
        }
        r2 = inner(R.data(), R.data(), Df1k);
        beta = r2/gamma;
    }
    UTX(X, m1, W1, P1);
    cout << "nr_cg: " << nr_cg << endl;
}

void ImpProblem::solve_side(const ImpInt &f1, const ImpInt &f2) {
    const ImpInt f12 = index_vec(f1, f2, f);
    Vec &W1 = W[f12], &H1 = H[f12], &P1 = P[f12], &Q1 = Q[f12];

    update_side(f1, f2, true);

    Vec G1(W1.size(), 0), G2(H1.size(), 0);

    gd_side(f1, Q1, G1);
    cg(f1, f2, W1, Q1, G1, P1);

    gd_side(f2, P1, G2);
    cg(f2, f1, H1, P1, G2, Q1);

    update_side(f1, f2, false);
}

void ImpProblem::solve_cross(const ImpInt &f1, const ImpInt &f2) {
    const ImpInt f12 = index_vec(f1, f2, f);
    Vec &W1 = W[f12], &H1 = H[f12], &P1 = P[f12], &Q1 = Q[f12];

    update_cross(f1, f2, true);

    Vec GW(W1.size()), GH(H1.size());

    gd_cross(f1, f12, Q1, GW);
    cg(f1, f2, W1, Q1, GW, P1);

    gd_cross(f2, f12, P1, GH);
    cg(f2, f1, H1, P1, GH, Q1);

    update_cross(f1, f2, false);
}

void ImpProblem::one_epoch() {

    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = f1; f2 < fu; f2++)
            solve_side(f1, f2);
    }

    for (ImpInt f1 = fu; f1 < f; f1++) {
        for (ImpInt f2 = f1; f2 < f; f2++) {
            solve_side(f1, f2);
        }
    }

    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = fu; f2 < f; f2++)
            solve_cross(f1, f2);
    }
    cache_sasb();
}

void ImpProblem::init_va(ImpInt size) {

    if (Ut->file_name.empty())
        return;

    mt = Ut->m;
    Pt.resize(fu*(f+f-fu)/2);
    Qt.resize(fu*(f+f-fu)/2);

    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            Pt[f12].resize(mt*k);
            Qt[f12].resize(mt*k);
        }
    }

    va_loss.resize(size);
    top_k.resize(size);
    ImpInt start = 5;

    cout << "iter";
    cout.width(12);
    cout << "loss";
    for (ImpInt i = 0; i < size; i++) {
        top_k[i] = start;
        cout.width(12);
        cout << "va_p@" << start;
        start *= 2;
    }
    cout << endl;
}


void ImpProblem::validate() {
    const ImpInt nr_th = param->nr_threads, nr_k = top_k.size();
    ImpLong valid_samples = 0;

    vector<ImpLong> hit_counts(nr_th*nr_k, 0);

   for (ImpInt f1 = 0; f1 < fu; f1++) {
       const vector<Node*> &X1 = U->Xs[f1];
       for (ImpInt f2 = f1; f2 < f; f2++) {
           const ImpInt f12 = index_vec(f1, f2, f);
            UTX(X1, Ut->m, W[f12], Pt[f12]);
            if (f2 < fu)
                UTX(X1, Ut->m, H[f12], Qt[f12]);
       }
   }

#pragma omp parallel for schedule(static) reduction(+: valid_samples)
    for (ImpLong i = 0; i < Ut->m; i++) {
        Vec z(n, MIN_Z);
        pred_z(i, z);
        prec_k(z, i, top_k, hit_counts);
        valid_samples++;
    }

    fill(va_loss.begin(), va_loss.end(), 0);

    for (ImpInt i = 0; i < nr_k; i++) {
        for (ImpLong num_th = 0; num_th < nr_th; num_th++) {
            va_loss[i] += hit_counts[i+num_th*nr_k];
        }
        va_loss[i] /= ImpDouble(valid_samples*top_k[i]);
    }
}

void ImpProblem::pred_z(ImpLong i, Vec &z) {
}

void ImpProblem::prec_k(Vec &z, ImpLong i, vector<ImpInt> &top_k, vector<ImpLong> &hit_counts) {
    ImpInt valid_count = 0;
    const ImpInt nr_k = top_k.size();
    vector<ImpLong> hit_count(nr_k, 0);

    ImpInt num_th = omp_get_thread_num();

#ifdef EBUG
    cout << i << ":";
#endif
    for (ImpInt state = 0; state < nr_k; state++) {
        while(valid_count < top_k[state]) {
            ImpLong argmax = distance(z.begin(), max_element(z.begin(), z.end()));
#ifdef EBUG
            cout << argmax << " ";
#endif
            z[argmax] = MIN_Z;

            for (Node* nd = Ut->Y[i]; nd < Ut->Y[i+1]; nd++) {
                if (argmax == nd->idx) {
                    hit_count[state]++;
                    break;
                }
            }
            valid_count++;
        }
    }

#ifdef EBUG
    cout << endl;
#endif
    for (ImpInt i = 1; i < nr_k; i++) {
        hit_count[i] += hit_count[i-1];
    }
    for (ImpInt i = 0; i < nr_k; i++) {
        hit_counts[i+num_th*nr_k] += hit_count[i];
    }
}

void ImpProblem::print_epoch_info(ImpInt t) {
    ImpInt nr_k = top_k.size();
    cout << "iter";
    cout.width(4);
    cout << t+1;
    if (!Ut->file_name.empty()) {
        for (ImpInt i = 0; i < nr_k; i++ ) {
            cout.width(13);
            cout << setprecision(3) << va_loss[i]*100;
        }
    }
    cout << endl;
} 

void ImpProblem::solve() {
    //init_va(4);
    for (ImpInt iter = 0; iter < param->nr_pass; iter++) {
        one_epoch();
        //validate();
        //print_epoch_info(iter);
    }
}
