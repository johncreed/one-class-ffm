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

void ImpProblem::UTX(shared_ptr<ImpData> &D, Vec &A, Vec &C) {
    fill(C.begin(), C.end(), 0);

    const vector<Node*> &X = D->X;
    const ImpLong l1 = D->l;
    ImpDouble* c = C.data();
    for (ImpLong i = 0; i < l1; i++)
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

    ImpLong idx, y_nnz=0, x_nnz=0;
    ImpDouble val;

    while (getline(fs, line)) {
        l++;
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

        while (iss >> idx >> dummy >> val) {
            m = max(m, idx+1);
            x_nnz++;
        }
    }

    m = min(m, max_m);

    fs.clear();
    fs.seekg(0);

    N.resize(x_nnz);

    X.resize(l+1);
    Y.resize(l+1);

    if (has_label) {
        M.resize(y_nnz);
    }

    vector<ImpInt> nnx(l, 0), nny(l, 0);

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

        while (iss >> idx >> dummy >> val) {
            if (idx >= m)
                continue;
            nnz_i++;
            N[nnz_i-1].idx = idx;
            N[nnz_i-1].val = val;
        }
        nnx[i] = nnz_i;
        i++;
    }

    X[0] = N.data();
    for (ImpLong i = 0; i < l; i++) {
        X[i+1] = N.data() + nnx[i];
    }

    if (has_label) {
        Y[0] = M.data();
        for (ImpLong i = 0; i < l; i++) {
            Y[i+1] = M.data() + nny[i];
        }
    }
    fs.close();
}

void ImpData::transY(const vector<Node*> &YT) {
    n = YT.size() - 1;
    vector<pair<ImpLong, Node*>> perm;
    ImpLong nnz = 0;
    vector<ImpLong> nnzs(l, 0);

    for (ImpLong i = 0; i < n; i++)
        for (Node* y = YT[i]; y < YT[i+1]; y++) {
            if (y->idx >= l )
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
    for (ImpLong nnz_i = 0; nnz_i < nnz; nnz_i++) {
        M[nnz_i].idx = perm[nnz_i].first;
        //M[nnz_i].val = perm[nnz_i].second->val;
    }

    Y[0] = M.data();
    ImpLong start_idx = 0;
    for (ImpLong i = 0; i < l; i++) {
        start_idx += nnzs[i];
        Y[i+1] = M.data()+start_idx;
    }
}

void ImpData::print_data_info() {
    cout << "File:";
    cout << file_name;
    cout.width(12);
    cout << "l:";
    cout << l;
    cout.width(12);
    cout << "m:";
    cout << m;
    cout << endl;
}


void ImpProblem::init() {
    k = param->k;
    lambda = param->lambda;
    w = param->eta;

    mc = Tr->m;
    mt = Xt->m;

    l = Tr->l;
    n = Xt->l;

    orders.resize(l);
    iota(orders.begin(), orders.end(), 0);

    init_mat(U, mc, k);
    init_mat(V, mt, k);

    CTC.resize(k*k, 0);

    P.resize(l*k, 0);
    Q.resize(n*k, 0);

    gu.resize(mc*k, 0);
    gv.resize(mt*k, 0);
}

void ImpProblem::gd(shared_ptr<ImpData> &data, Vec &A, Vec &C, Vec &D, Vec &G) {
    const ImpLong l = data->l, m = data->m, mk = m*k;
    const vector<Node*> &X = data->X, &Y = data->Y;

    Vec T(l*k, 0);

    ImpDouble *cx = C.data(), *dx = D.data(), *ctc = CTC.data(), *tx = T.data();
    // calc P(QTQ)
    mm(cx, ctc, tx, l, k, k);

    for (ImpLong jd = 0; jd < mk; jd++)
        G[jd] = lambda*A[jd];


    for (ImpLong i = 0; i < l; i++) {
        const ImpDouble *cp = cx + i*k; 
        const ImpDouble *tp = tx + i*k;

        Vec phi(k, 0);

        for (Node *y = Y[i]; y < Y[i+1]; y++) {
            const ImpLong idx = y->idx;
            const ImpDouble *dp = dx + idx*k;
            const ImpDouble scale = (1-w)*inner(cp, dp, k)-1;
            for (ImpInt d = 0; d < k; d++)
                phi[d] += scale*dp[d];
        }

        for (Node *x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++) {
                const ImpLong jd = idx*k+d;
                G[jd] += val*(phi[d]+w*tp[d]);
            }
        }
    }
    cout << "GD: "<< inner(G.data(), G.data(), mk) << endl;
}

void ImpProblem::Hs(shared_ptr<ImpData> &data, Vec &A, Vec &D, Vec &S, Vec &HS) {
    const ImpLong l = data->l, m = data->m, mk = m*k;
    const vector<Node*> &X = data->X, &Y = data->Y;
    const ImpDouble *ctc = CTC.data(), *dx = D.data();

    Vec T(mk, 0);

    ImpDouble *tx = T.data(), *sx = S.data();

    for (ImpLong jd = 0; jd < mk; jd++)
        HS[jd] = lambda*S[jd];

    mm(sx, ctc, tx, m, k, k);

    for (ImpLong i = 0; i < l; i++) {
        Vec tau(k, 0), phi(k, 0), ka(k, 0);

        UTx(X[i], X[i+1], S, phi.data());
        UTx(X[i], X[i+1], T, tau.data());

        for (Node* y = Y[i]; y < Y[i+1]; y++) {
            const ImpLong idx = y->idx;
            const ImpDouble *dp = dx + idx*k;
            const ImpDouble val = inner(phi.data(), dp, k);
            for (ImpInt d1 = 0; d1 < k; d1++)
                ka[d1] += val*dp[d1];
        }

        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d1 = 0; d1 < k; d1++) {
                const ImpLong jd = idx*k+d1;
                HS[jd] += ((1-w)*phi[d1]+w*tau[d1])*val;
            }
        }
    }
}

void ImpProblem::cg(shared_ptr<ImpData> &data, Vec &A, Vec &D, Vec &G) {
    const ImpLong m = data->m, mk = m*k;

    ImpInt nr_cg = 0 , max_cg = 25;
    ImpDouble g2 = 0, r2, cg_eps = 1e-5, alpha = 0, beta = 0, gamma = 0, sHs;

    Vec S(mk, 0), R(mk, 0), HS(mk, 0);

    for (ImpLong jd = 0; jd < mk; jd++) {
        R[jd] = -G[jd];
        S[jd] = R[jd];
        g2 += G[jd]*G[jd];
    }

    r2 = g2;

    while (g2*cg_eps < r2 && nr_cg < max_cg) {
        nr_cg++;
        for (ImpLong jd = 0; jd < mk; jd++)
            S[jd] = R[jd] + beta*S[jd];

        Hs(data, A, D, S, HS);

        sHs = inner(S.data(), HS.data(), mk);
        gamma = r2;
        alpha = gamma/sHs;
        for (ImpLong jd = 0; jd < mk; jd++) {
            A[jd] += alpha*S[jd];
            R[jd] -= alpha*HS[jd];
        }
        r2 = inner(R.data(), R.data(), mk);
        beta = r2/gamma;
    }
    cout << "nr_cg: " << nr_cg << endl;
}

void ImpProblem::func(){
  UTX(Tr, U, P);
  UTX(Xt, V, Q);
  vector<Node*> &Y = Tr->Y;
  ImpDouble func_val = 0;
  for( ImpLong i = 0; i < l; i++){
      ImpDouble *Pp = P.data() + i*k;
      for( ImpLong j = 0; j < n; j++){
        ImpDouble *Qp = Q.data() + j*k;
        ImpDouble YHat = inner(Pp, Qp, k);
        bool check = false;
        for( Node* y = Y[i]; y < Y[i+1]; y++){
          if ( j == y->idx ){
            check = true;
            break;
          }
        }
        ImpDouble YVal = (check)? 1 : 0;
        ImpDouble CIJ = (check)? 1 : w;
        func_val += 0.5 * CIJ * (YVal - YHat) * (YVal - YHat);
      }
  }
  func_val += 0.5 * lambda * inner(V.data(), V.data(), V.size());
  func_val += 0.5 * lambda * inner(U.data(), U.data(), U.size());
/*    ImpDouble f = 0;
    for(ImpLong i = 0 ; i < l ; i++)
        for(Node* y = Y[i]; y < Y[i+1]; y++)
          f += 0.5;
    f += 0.5 * lambda * inner(V.data(), V.data(), V.size());
    f += inner(G.data(), A.data(), mk);
    Hs(data, A, D, A, HS);
    f += 0.5 * inner( A.data(), HS.data(), mk);
    cout << "fnc_val: " << f << endl;
    */
  cout << "func_val: " << func_val << endl;
}


void ImpProblem::one_epoch() {

    func();
    QTQ(Q, n);
    gd(Tr, U, P, Q, gu);
    cg(Tr, U, Q, gu);
    UTX(Tr, U, P);
    //gd(Tr, U, P, Q, gu);

    QTQ(P, l);
    gd(Xt, V, Q, P, gv);
    cg(Xt, V, P, gv);
    UTX(Xt, V, Q);
    //gd(Xt, V, Q, P, gv);
}

void ImpProblem::init_va_loss(ImpInt size) {

    if (Te->file_name.empty())
        return;
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

    UTX(Xt, V, Q);

#pragma omp parallel for schedule(static) reduction(+: valid_samples)
    for (ImpLong i = 0; i < Te->l; i++) {
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
    Vec pct(k, 0);
    ImpDouble *p = pct.data(), *qx = Q.data();
    UTx(Te->X[i], Te->X[i+1], U, p);
    for (ImpLong j = 0; j < n; j++) {
        const ImpDouble *q = qx+j*k;
        z[j] = inner(p, q, k);
    }
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

            for (Node* nd = Te->Y[i]; nd < Te->Y[i+1]; nd++) {
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
    if (!Te->file_name.empty()) {
        for (ImpInt i = 0; i < nr_k; i++ ) {
            cout.width(13);
            cout << setprecision(3) << va_loss[i]*100;
        }
    }
    cout << endl;
} 

void ImpProblem::solve() {
    init_va_loss(4);
    UTX(Tr, U, P);
    UTX(Xt, V, Q);
    for (ImpInt iter = 0; iter < param->nr_pass; iter++) {
        one_epoch();
        //validate();
        print_epoch_info(iter);
    }
}
