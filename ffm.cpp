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

void mm(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong l, const ImpLong n, const ImpInt k) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            l, n, k, 1, a, k, b, n, 0, c, n);
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

void ImpProblem::UTx(Node* x0, Node* x1, Vec &A, ImpDouble *c) {
    for (Node* x = x0; x < x1; x++) {
        const ImpLong idx = x->idx;
        const ImpDouble val = x->val;
        for (ImpInt d = 0; d < k; d++) {
            ImpLong jd = idx*k+d;
            c[d] += val*A[jd];
        }
    }
}

void ImpProblem::UTX(const vector<Node*> &X, ImpLong m1, Vec &A, Vec &C) {
    fill(C.begin(), C.end(), 0);

    ImpDouble* c = C.data();
    for (ImpLong i = 0; i < m1; i++)
        UTx(X[i], X[i+1], A, c+i*k);
}

void ImpProblem::QTQ(const Vec &C, const ImpLong &l1) {
    const ImpDouble *c = C.data();
    ImpDouble *ctc = CTC.data();
    fill(CTC.begin(), CTC.end(), 0);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            k, k, l1, 1, c, k, c, k, 0, ctc, k);
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

    m = U->m;
    n = V->m;

    fu = U->f;
    fv = V->f;
    f = fu+fv;

    k = param->k;

    a.resize(m, 0);
    b.resize(n, 0);
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

    calc_side();
    init_y_tilde();
}

void ImpProblem::solve_side(const ImpInt &f1, const ImpInt &f2) {
    const ImpInt f12 = index_vec(f1, f2, f);

    const ImpInt base = (f1 < fu)? 0: fu;
    const ImpInt fi = f1-base, fj = f2-base;

    shared_ptr<ImpData> U1 = (f1 < fu)? U:V;
    vector<Node*> &UX = U1->Xs[fi],  &VX = U1->Xs[fj];

    Vec &W1 = W[f12], &H1 = H[f12], &P1 = P[f12], &Q1 = Q[f12];
    const ImpLong Df1 = U1->Ds[fi], Df2 = U1->Ds[fj];


    update_side(f1, f2, false);

    //GD
    //CG
    UTX(UX, U1->m, W1, P1);

    //GD
    //CG
    UTX(VX, U1->m, H1, Q1);

    update_side(f1, f2, true);
}

void ImpProblem::solve_cross(const ImpInt &f1, const ImpInt &f2) {
    const ImpInt f12 = index_vec(f1, f2, f);
    Vec &W1 = W[f12], &H1 = H[f12], &P1 = P[f12], &Q1 = Q[f12];

    const ImpInt fi = f1, fj = f2-fu;
    vector<Node*> &UX = U->Xs[fi], &VX = V->Xs[fj];
    const ImpLong Df1 = U->Ds[fi], Df2 = V->Ds[fj];

    update_cross(f1, f2, false);

    //GD
    //CG
    UTX(UX, U->m, W1, P1);

    //GD
    //CG
    UTX(VX, V->m, H1, Q1);

    update_cross(f1, f2, true);

}

void ImpProblem::one_epoch() {
    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = f1; f2 < fu; f2++)
            solve_side(f1, f2);
        for (ImpInt f2 = fu; f2 < f; f2++)
            solve_cross(f1, f2);
    }
    for (ImpInt f1 = fu; f1 < f; f1++) {
        for (ImpInt f2 = f1; f2 < f; f2++) {
            solve_side(f1, f2);
        }
    }
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
    init();
    //init_va(4);
    for (ImpInt iter = 0; iter < param->nr_pass; iter++) {
        one_epoch();
        //validate();
        //print_epoch_info(iter);
    }
}
