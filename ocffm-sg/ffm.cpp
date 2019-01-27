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

void scal(ImpDouble *x, const ImpLong &l, const ImpDouble &lambda) {
    cblas_dscal(l, lambda, x, 1);
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
        const ImpLong k, const ImpLong l) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            k, k, l, 1, a, k, b, k, 0, c, k);
}

void mv(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong l, const ImpInt k, const ImpDouble &beta, bool trans) {
    const CBLAS_TRANSPOSE CBTr= (trans)? CblasTrans: CblasNoTrans;
    cblas_dgemv(CblasRowMajor, CBTr, l, k, 1, a, k, b, 1, beta, c, 1);
}

const ImpInt index_vec(const ImpInt f1, const ImpInt f2, const ImpInt f) {
    assert( f1 <= f2);
    return f2 + (f-1)*f1 - f1*(f1-1)/2;
}

ImpDouble inner(const ImpDouble *p, const ImpDouble *q, const ImpInt k)
{
    return cblas_ddot(k, p, 1, q, 1);
}

void row_wise_inner(const Vec &V1, const Vec &V2, const ImpLong &row,
        const ImpLong &col,const ImpDouble &alpha, Vec &vv){
    const ImpDouble *v1p = V1.data(), *v2p = V2.data();

    #pragma omp parallel for schedule(guided)
    for(ImpInt i = 0; i < row; i++)
        vv[i] += alpha*inner(v1p+i*col, v2p+i*col, col);
}

void init_mat(Vec &vec, const ImpLong nr_rows, const ImpLong nr_cols) {
    default_random_engine ENGINE(rand());
    vec.resize(nr_rows*nr_cols, 0.1);
    uniform_real_distribution<ImpDouble> dist(-0.1*qrsqrt(nr_cols), 0.1*qrsqrt(nr_cols));

    auto gen = std::bind(dist, ENGINE);
    generate(vec.begin(), vec.end(), gen);
}

void ImpData::read(bool has_label, const ImpLong *ds) {
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
            if (ds!= nullptr && ds[fid] <= idx)
                continue;
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
        nnz_y = y_nnz;
        M.resize(y_nnz);
        popular.resize(n);
        fill(popular.begin(), popular.end(), 0);
        obs_sets.resize(m);
    }

    nnx.resize(m);
    nny.resize(m);
    fill(nnx.begin(), nnx.end(), 0);
    fill(nny.begin(), nny.end(), 0);

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
                popular[idx] += 1;
                obs_sets[i].insert(idx);
            }
            nny[i] = nnz_j;
        }

        while (iss >> fid >> dummy >> idx >> dummy >> val) {
            if (ds!= nullptr && ds[fid] <= idx)
                continue;
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

    ImpDouble sum = 0;
    for (auto &n : popular)
        sum += n;
    for (auto &n : popular)
        n /= sum;

    for (ImpLong i = m-1; i > 0; i--) {
        nnx[i] -= nnx[i-1];
        nny[i] -= nny[i-1];
    }
    fs.close();
}

void ImpData::split_fields() {
    Ns.resize(f);
    Xs.resize(f);
    Ds.resize(f);
    freq.resize(f);


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

    for(ImpInt fi = 0; fi < f; fi++){
        freq[fi].resize(Ds[fi]);
        fill(freq[fi].begin(), freq[fi].end(), 0);
    }

    for( ImpLong i = 0; i < m; i++){
        for(Node* x = X[i]; x < X[i+1]; x++){
            ImpInt fid = x->fid;
            ImpLong idx = x->idx;
            freq[fid][idx]++;
        }
    }
    for (ImpInt fi = 0; fi < f; fi++) {
        Node* fM = Ns[fi].data();
        Xs[fi][0] = fM;
        ImpLong start = 0;
        for (ImpLong i = 0; i < m; i++) {
            Xs[fi][i+1] = fM + start + f_nnz[fi][i];
            start += f_nnz[fi][i];
        }
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
        M[nnz_i].val = perm[nnz_i].second->val;
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
    cout << "d:";
    cout << Ds[0];
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
#pragma omp parallel for schedule(guided)
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
    GW_sum[f12].resize(Df1*k, 1);
    GH_sum[f12].resize(Df2*k, 1);
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
    for (ImpInt f1 = fu; f1 < f; f1++) {
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            add_side(P[f12], Q[f12], n, b);
        }
    }
}

ImpDouble ImpProblem::calc_cross(const ImpLong &i, const ImpLong &j) {
    ImpDouble cross_value = 0.0;
    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = fu; f2 < f; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            const ImpDouble *pp = P[f12].data();
            const ImpDouble *qp = Q[f12].data();
            cross_value += inner(pp+i*k, qp+j*k, k);
        }
    }
    return cross_value;
}

void ImpProblem::init_y_tilde() {
    #pragma omp parallel for schedule(guided)
    for (ImpLong i = 0; i < m; i++) {
        for (Node* y = U->Y[i]; y < U->Y[i+1]; y++) {
            ImpLong j = y->idx;
            y->val = a[i]+b[j]+calc_cross(i, j) - 1;
        }
    }
    #pragma omp parallel for schedule(guided)
    for (ImpLong j = 0; j < n; j++) {
        for (Node* y = V->Y[j]; y < V->Y[j+1]; y++) {
            ImpLong i = y->idx;
            y->val = a[i]+b[j]+calc_cross(i, j) - 1;
        }
    }
}


void ImpProblem::init() {
    lambda = param->lambda;
    w = param->omega;
    r = param->r;
    eta = param->eta;

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

    const ImpInt nr_blocks = f*(f+1)/2;

    W.resize(nr_blocks);
    H.resize(nr_blocks);
    GW_sum.resize(nr_blocks);
    GH_sum.resize(nr_blocks);

    P.resize(nr_blocks);
    Q.resize(nr_blocks);

    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? U: V);
        const ImpInt fi = ((f1>=fu)? f1-fu: f1);
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? U: V);
            const ImpInt fj = ((f2>=fu)? f2-fu: f2);
            const ImpInt f12 = index_vec(f1, f2, f);
            if(!param->self_side && (f1>=fu || f2<fu))
                continue;
            init_pair(f12, fi, fj, d1, d2);
        }
    }

    cache_sasb();
    if (param->self_side)
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

void ImpProblem::one_epoch() {

    cout << m * n << endl;
    cout << m << " " <<  n << endl;
    ImpLong counter = 0;
    vector<ImpLong> outer_order(m);
    iota(outer_order.begin(), outer_order.end(), 0);
    //random_shuffle(outer_order.begin(), outer_order.end());
    #pragma omp parallel for schedule(guided) reduction(+: counter)
    for(ImpLong ii = 0; ii < outer_order.size(); ii++){
        ImpLong i = outer_order[ii];
        vector<ImpLong> inner_order(n);
        iota(inner_order.begin(), inner_order.end(), 0);
        //random_shuffle(inner_order.begin(), inner_order.end());
        for(auto j: inner_order){
            update_p_q(i, j);
            update_W_H(i, j);
            if( counter % 100000 == 0)
                cout << counter << " " << flush   ;
            counter ++;
        }
    }

}

void ImpProblem::init_va(ImpInt size) {

    if (Uva->file_name.empty())
        return;

    mt = Uva->m;

    const ImpInt nr_blocks = f*(f+1)/2;

    Pva.resize(nr_blocks);
    Qva.resize(nr_blocks);

    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? Uva: V);
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? Uva: V);
            const ImpInt f12 = index_vec(f1, f2, f);
            if(!param->self_side && (f1>=fu || f2<fu))
                continue;
            Pva[f12].resize(d1->m*k);
            Qva[f12].resize(d2->m*k);
        }
    }

    va_loss_prec.resize(size);
    va_loss_ndcg.resize(size);
    top_k.resize(size);
    ImpInt start = 5;

    cout << "iter";
    for (ImpInt i = 0; i < size; i++) {
        top_k[i] = start;
        cout.width(9);
        cout << "( p@ " << start << ", ";
        cout.width(6);
        cout << "nDCG@" << start << " )";
        start *= 2;
    }
    cout.width(12);
    cout << "ploss";
    cout << endl;
}

void ImpProblem::pred_z(const ImpLong i, ImpDouble *z) {
    for(ImpInt f1 = 0; f1 < fu; f1++) {
        for(ImpInt f2 = fu; f2 < f; f2++) {
            ImpInt f12 = index_vec(f1, f2, f);
            ImpDouble *p1 = Pva[f12].data()+i*k, *q1 = Qva[f12].data();
            mv(q1, p1, z, n, k, 1, false);
        }
    }
}

void ImpProblem::validate() {
    const ImpInt nr_th = param->nr_threads, nr_k = top_k.size();
    ImpLong valid_samples = 0;

    vector<ImpLong> hit_counts(nr_th*nr_k, 0);
    vector<ImpDouble> ndcg_scores(nr_th*nr_k, 0);

    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? Uva: V);
        const ImpInt fi = ((f1>=fu)? f1-fu: f1);

        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? Uva: V);
            const ImpInt fj = ((f2>=fu)? f2-fu: f2);

            const ImpInt f12 = index_vec(f1, f2, f);
            if(!param->self_side && (f1>=fu || f2<fu))
                continue;
            UTX(d1->Xs[fi], d1->m, W[f12], Pva[f12]);
            UTX(d2->Xs[fj], d2->m, H[f12], Qva[f12]);
        }
    }

    Vec at(Uva->m, 0), bt(V->m, 0);

    if (param->self_side) {
        for (ImpInt f1 = 0; f1 < fu; f1++) {
            for (ImpInt f2 = f1; f2 < fu; f2++) {
                const ImpInt f12 = index_vec(f1, f2, f);
                add_side(Pva[f12], Qva[f12], Uva->m, at);
            }
        }
        for (ImpInt f1 = fu; f1 < f; f1++) {
            for (ImpInt f2 = f1; f2 < f; f2++) {
                const ImpInt f12 = index_vec(f1, f2, f);
                add_side(Pva[f12], Qva[f12], V->m, bt);
            }
        }
    }

    ImpDouble ploss = 0;
#ifdef EBUG
    for (ImpLong i = 0; i < n; i++) {
        cout << U->popular[i] << " ";
    }
    cout << endl;
#endif
#pragma omp parallel for schedule(static) reduction(+: valid_samples, ploss)
    for (ImpLong i = 0; i < Uva->m; i++) {
        Vec z, z_copy;
        if(Uva->nnx[i] == 0) {
            z.assign(U->popular.begin(), U->popular.end());
        }
        else {
            z.assign(bt.begin(), bt.end());
            pred_z(i, z.data());
        }
        for(Node* y = Uva->Y[i]; y < Uva->Y[i+1]; y++){
            const ImpLong j = y->idx;
            if (j < z.size())
                ploss += (1-z[j]-at[i])*(1-z[j]-at[i]);
        }

#ifdef EBUG_nDCG
        z.resize(n);
        z_copy.resize(n);
        for(ImpInt i = 0; i < n ; i++)
          z[i] = z_copy[i] = n - i;
#endif
        z_copy.assign(z.begin(), z.end());
        // Precision @
        prec_k(z.data(), i, hit_counts);
        // nDCG
        ndcg(z_copy.data(), i, ndcg_scores);
        valid_samples++;
    }

    loss = sqrt(ploss/Uva->m);

    fill(va_loss_prec.begin(), va_loss_prec.end(), 0);
    fill(va_loss_ndcg.begin(), va_loss_ndcg.end(), 0);
    for (ImpInt i = 0; i < nr_k; i++) {

        for (ImpLong num_th = 0; num_th < nr_th; num_th++){
            va_loss_prec[i] += hit_counts[i+num_th*nr_k];
            va_loss_ndcg[i] += ndcg_scores[i+num_th*nr_k];
        }

        va_loss_prec[i] /= ImpDouble(valid_samples*top_k[i]);
        va_loss_ndcg[i] /= ImpDouble(valid_samples);
    }
}

void ImpProblem::prec_k(ImpDouble *z, ImpLong i, vector<ImpLong> &hit_counts) {
    ImpInt valid_count = 0;
    const ImpInt nr_k = top_k.size();
    vector<ImpLong> hit_count(nr_k, 0);

    ImpInt num_th = omp_get_thread_num();

#ifdef EBUG
    //cout << i << ":";
#endif
    ImpLong max_z_idx = U->popular.size();
    for (ImpInt state = 0; state < nr_k; state++) {
        while(valid_count < top_k[state]) {
            if ( valid_count >= max_z_idx )
               break;
            ImpLong argmax = distance(z, max_element(z, z + max_z_idx));
#ifdef EBUG
    //        cout << argmax << " ";
#endif
            z[argmax] = MIN_Z;
            for (Node* nd = Uva->Y[i]; nd < Uva->Y[i+1]; nd++) {
                if (argmax == nd->idx) {
                    hit_count[state]++;
                    break;
                }
            }
            valid_count++;
        }
    }

#ifdef EBUG
    //cout << endl;
#endif
    for (ImpInt i = 1; i < nr_k; i++) {
        hit_count[i] += hit_count[i-1];
    }
    for (ImpInt i = 0; i < nr_k; i++) {
        hit_counts[i+num_th*nr_k] += hit_count[i];
    }
}

void ImpProblem::ndcg(ImpDouble *z, ImpLong i, vector<ImpDouble> &ndcg_scores) {
    ImpInt valid_count = 0;
    const ImpInt nr_k = top_k.size();
    vector<ImpDouble> dcg_score(nr_k, 0);
    vector<ImpDouble> idcg_score(nr_k, 0);

    ImpInt num_th = omp_get_thread_num();

#ifdef EBUG_nDCG
#ifndef SHOW_SCORE_ONLY
    bool show_label = true;
    cout << i << ":";
#endif
#endif
    ImpLong max_z_idx = U->popular.size();
    for (ImpInt state = 0; state < nr_k; state++) {
        while(valid_count < top_k[state]) {
            if ( valid_count >= max_z_idx )
               break;
            ImpLong argmax = distance(z, max_element(z, z + max_z_idx));
#ifdef EBUG_nDCG
#ifndef SHOW_SCORE_ONLY
            if( 10 < top_k[state] )
              break;
            cout << argmax << " ";
#endif
#endif
            z[argmax] = MIN_Z;
#ifdef EBUG_nDCG
#ifndef SHOW_SCORE_ONLY
            if(show_label) {
              cout << "(";
              for (Node* nd = Uva->Y[i]; nd < Uva->Y[i+1]; nd++)
                  cout << nd->idx << ",";
              cout << ")" << endl;
              show_label = false;
            }
#endif
#endif
            for (Node* nd = Uva->Y[i]; nd < Uva->Y[i+1]; nd++) {
                if (argmax == nd->idx) {
                    dcg_score[state] += 1.0 / log2(valid_count + 2);
                    break;
                }
            }

            if( ImpInt(Uva->Y[i+1] - Uva->Y[i]) > valid_count )
                idcg_score[state] += 1.0 / log2(valid_count + 2);
            valid_count++;
        }
    }

#ifdef EBUG_nDCG
#ifndef SHOW_SCORE_ONLY
    cout << endl;
    cout << dcg_score[0] << ", " << idcg_score[0] << endl;
#endif
#endif
    for (ImpInt i = 1; i < nr_k; i++) {
        dcg_score[i] += dcg_score[i-1];
        idcg_score[i] += idcg_score[i-1];
    }

    for (ImpInt i = 0; i < nr_k; i++) {
        ndcg_scores[i+num_th*nr_k] += dcg_score[i] / idcg_score[i];
    }
#ifdef EBUG_nDCG
    cout << setprecision(4) << dcg_score[1] / idcg_score[1] << endl;
#endif
}

void ImpProblem::print_epoch_info(ImpInt t) {
    ImpInt nr_k = top_k.size();
    cout.width(2);
    cout << t+1;
    if (!Uva->file_name.empty()) {
        for (ImpInt i = 0; i < nr_k; i++ ) {
            cout.width(9);
            cout << "( " <<setprecision(3) << va_loss_prec[i]*100 << " ,";
            cout.width(6);
            cout << setprecision(3) << va_loss_ndcg[i]*100 << " )";
        }
        cout.width(13);
        cout << setprecision(3) << loss;
    }
    cout << endl;
}

void ImpProblem::solve() {
    init_va(5);
    for (ImpInt iter = 0; iter < param->nr_pass; iter++) {
            one_epoch();
            if (!Uva->file_name.empty() && iter % 1 == 0) {
                update_P_Q();
                validate();
                print_epoch_info(iter);
            }
    }
}

void ImpProblem::write_header(ofstream &f_out) const{
    f_out << f << endl;
    f_out << fu << endl;
    f_out << fv << endl;
    f_out << k << endl;
    
    for(ImpInt fi = 0; fi < fu ; fi++)
        f_out << U->Ds[fi] << endl;
    
    for(ImpInt fi = 0; fi < fv ; fi++)
        f_out << V->Ds[fi] << endl;
}

void write_block(const Vec& block, const ImpLong& num_of_rows, const ImpInt& num_of_columns, char block_type, const ImpInt fi, const ImpInt fj, ofstream &f_out){
#ifdef DEBUG_SAVE
    if ( block.size() != num_of_columns * num_of_rows ){
        cout << block.size() << " " << num_of_columns << " " << num_of_rows << endl;
        assert(false);
    }
#endif
    ostringstream stringStream;
    stringStream << block_type << ',' << fi << ',' << fj;
    string line_info =  stringStream.str();

    for( ImpLong row_i = 0; row_i < num_of_rows; row_i++ ){
        f_out << line_info << ',' << row_i;
        ImpLong offset = row_i * num_of_columns;
        for(ImpInt col_i = 0; col_i < num_of_columns ; col_i++ ){
            f_out << " " <<block[offset + col_i];
        }
        f_out << endl;
    }
}

void ImpProblem::write_W_and_H(ofstream &f_out) const{
    for(ImpInt fi = 0; fi < f ; fi++){
        for(ImpInt fj = fi; fj < f; fj++){
            ImpInt fij = index_vec(fi, fj, f);
            ImpInt fi_base = (fi >= fu )? fi - fu : fi;
            ImpInt fj_base = (fj >= fu )? fj - fu : fj;
            if ( fi < fu && fj < fu ){
                if( !param->self_side )
                    continue;
                write_block(W[fij], U->Ds[fi_base], k, 'W', fi, fj, f_out);
                write_block(H[fij], U->Ds[fj_base], k, 'H', fi, fj, f_out);
            }
            else if (fi < fu && fj >= fu){
                write_block(W[fij], U->Ds[fi_base], k, 'W', fi, fj, f_out);
                write_block(H[fij], V->Ds[fj_base], k, 'H', fi, fj, f_out);
            }
            else if( fi >= fu && fj >= fu){
                if( !param->self_side )
                    continue;
                write_block(W[fij], V->Ds[fi_base], k, 'W', fi, fj, f_out);
                write_block(H[fij], V->Ds[fj_base], k, 'H', fi, fj, f_out);
            }
        }
    }

}

void save_model(const ImpProblem& prob, string & model_path ){
#ifdef DEBUG_SAVE
    cout << "Start a save\n" << flush;
#endif
    ofstream f_out(model_path, ios::out | ios::trunc );
#ifdef DEBUG_SAVE
    cout << "Success open file.\n" << flush;
#endif
    prob.write_header( f_out );
#ifdef DEBUG_SAVE
    cout << "Success write header.\n" << flush;
#endif
    prob.write_W_and_H( f_out );  
}

ImpDouble ImpProblem::pq(const ImpInt &i, const ImpInt &j,const ImpInt &f1, const ImpInt &f2) {
    ImpInt f12 = index_vec(f1, f2, f);
    ImpInt Pi = (f1 < fu)? i : j;
    ImpInt Qj = (f2 < fu)? i : j;
    ImpDouble  *pp = P[f12].data()+Pi*k, *qp = Q[f12].data()+Qj*k;
    return inner(qp, pp, k);
}

ImpDouble ImpProblem::norm_block(const ImpInt &f1,const ImpInt &f2) {
    ImpInt f12 = index_vec(f1, f2, f);
    Vec &W1 = W[f12], H1 = H[f12];
    ImpDouble res = 0;
    res += inner(W1.data(), W1.data(), W1.size());
    res += inner(H1.data(), H1.data(), H1.size());
    return res;
}


ImpDouble ImpProblem::func() {
    ImpDouble res = 0;
    for (ImpInt i = 0; i < m; i++) {
        for(ImpInt j = 0; j < n; j++){
            ImpDouble y_hat = 0;
            for(ImpInt f1 = 0; f1 < f; f1++){
                for(ImpInt f2 = f1; f2 < f; f2++){
                    y_hat += pq(i, j, f1, f2);
                }
            }
            bool pos_term = false;
            for(Node* y = U->Y[i]; y < U->Y[i+1]; y++){
                if ( y->idx == j ) {
                    pos_term = true;
                    break;
                }
            }
            if( pos_term )
                res += (1 - y_hat) * (1 - y_hat);
            else
                res += w * (r - y_hat) * (r - y_hat);
        }
    }

    for(ImpInt f1 = 0; f1 < f; f1++) {
        for(ImpInt f2 = f1; f2 < f; f2++){
            res += lambda*norm_block(f1, f2);
        }
    }
    return 0.5*res;
}

void ImpProblem::update_P_Q(){
    #pragma omp parallel for schedule(guided)
    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? U: V);
        const ImpInt fi = ((f1>=fu)? f1-fu: f1);
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? U: V);
            const ImpInt fj = ((f2>=fu)? f2-fu: f2);
            const ImpInt f12 = index_vec(f1, f2, f);
            if(!param->self_side && (f1>=fu || f2<fu))
                continue;
            const vector<Node*> &X1 = d1->Xs[fi], &X2 = d2->Xs[fj];
            UTX(X1, d1->m, W[f12], P[f12]);
            UTX(X2, d2->m, H[f12], Q[f12]);
        }
    }
}

void ImpProblem::update_p_q(ImpLong i, ImpLong j){
    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? U: V);
        const ImpInt fi = ((f1>=fu)? f1-fu: f1);
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? U: V);
            const ImpInt fj = ((f2>=fu)? f2-fu: f2);
            const ImpInt f12 = index_vec(f1, f2, f);
            if(!param->self_side && (f1>=fu || f2<fu))
                continue;
            const vector<Node*> &X1 = d1->Xs[fi], &X2 = d2->Xs[fj];
            ImpInt Pi = (f1 < fu)? i : j;
            ImpInt Qj = (f2 < fu)? i : j;
            fill(P[f12].data() + Pi*k, P[f12].data() + Pi*k + k, 0);
            fill(Q[f12].data() + Qj*k, Q[f12].data() + Qj*k + k, 0);
            UTx(X1[Pi], X1[Pi+1], W[f12], P[f12].data() + Pi * k );
            UTx(X2[Qj], X2[Qj+1], H[f12], Q[f12].data() + Qj * k );
        }
    }
}

void ImpProblem::update_W_H(ImpLong i, ImpLong j){
    ImpDouble Y_hat = 0;
    for(ImpInt f1 = 0; f1 < f; f1++){
        for(ImpInt f2 = f1; f2 < f; f2++){
            Y_hat += pq(i, j, f1, f2);
        }
    }
    bool is_obs = ( U->obs_sets[i].find(j) != U->obs_sets[i].end() )? true : false;
    ImpDouble loss_grad = (is_obs)? -(1.0 - Y_hat) : - w * (r - Y_hat);
    for(ImpLong f1 = 0; f1 < f; f1++){
        const shared_ptr<ImpData> d = ((f1<fu)? U: V);
        const ImpLong f1_base = (f1 < fu)? f1 : f1 - fu;
#ifdef DEBUG
        assert( d->freq.size() > f1_base );
        assert( d->Xs.size() > f1_base );
        if(f1<fu)
            assert(d->Xs[f1_base].size() > i+1);
        else
            assert(d->Xs[f1_base].size() > j+1);
#endif
        const vector<ImpLong> &freq_ = d->freq[f1_base];
        Node *X_begin = (f1<fu)?  d->Xs[f1_base][i] : d->Xs[f1_base][j];
        Node *X_end = (f1<fu)?  d->Xs[f1_base][i+1] : d->Xs[f1_base][j+1];
        //Solve W f1 f2
        for(ImpLong f2 = f1; f2 < f; f2++){
            const ImpInt f12 = index_vec(f1, f2, f);
#ifdef DEBUG
            assert(Q.size() > f12);
            if(f2<fu)
                assert(Q[f12].size() > i*k);
            else
                assert(Q[f12].size() > j*k);
            assert(W.size() > f12);
            assert(GW_sum.size() > f12);
#endif
            ImpDouble *qp = (f2<fu)? (Q[f12].data() + i * k) : (Q[f12].data() + j * k);
            Vec &W12 = W[f12], &GW_sum12 = GW_sum[f12];
            for(Node *x = X_begin; x != X_end; x++){
#ifdef DEBUG
                assert(W12.size() > x->idx * k);
                assert(GW_sum12.size() > x->idx * k);
#endif
                ImpDouble *w = W12.data() + x->idx * k;
                ImpDouble *Gw = GW_sum12.data() + (x->idx * k);
                Vec gw(k, 0);
#ifdef DEBUG
                assert(freq_.size() > x->idx);
#endif
                ImpDouble lambda_freq = lambda / (ImpDouble) freq_[x->idx];
                axpy(w, gw.data(), k, lambda_freq);
                axpy(qp, gw.data(), k, loss_grad * x->val);
                for(ImpLong d = 0; d < k; d++){
                    Gw[d] += gw[d] * gw[d];
                    w[d] -= eta / sqrt(Gw[d]) * gw[d];
                }
            }
        }
        //Solve H f1 f2
        for(ImpLong f2 = 0; f2 < f1; f2++){
            const ImpInt f12 = index_vec(f2, f1, f);
#ifdef DEBUG
            assert(P.size() > f12);
            if(f2<fu)
                assert(P[f12].size() > i*k);
            else
                assert(P[f12].size() > j*k);
            assert(H.size() > f12);
            assert(GH_sum.size() > f12);
#endif
            ImpDouble *pp = (f2<fu)? (P[f12].data() + i * k) : (P[f12].data() + j * k);
            Vec &H12 = H[f12], &GH_sum12 = GH_sum[f12];
            for(Node *x = X_begin; x != X_end; x++){
#ifdef DEBUG
                assert(H12.size() > x->idx * k);
                assert(GH_sum12.size() > x->idx * k);
#endif
                ImpDouble *h = H12.data() + x->idx * k;
                ImpDouble *Gh = GH_sum12.data() + x->idx * k;
                Vec gh(k, 0);
#ifdef DEBUG
                assert(freq_.size() > x->idx);
#endif
                ImpDouble lambda_freq = lambda / (ImpDouble) freq_[x->idx];
                axpy(h, gh.data(), k, lambda_freq);
                axpy(pp, gh.data(), k, loss_grad * x->val);
                for(ImpLong d = 0; d < k; d++){
                    Gh[d] += gh[d] * gh[d];
                    h[d] -= eta / sqrt(Gh[d]) * gh[d];
                }
            }
        }
    }
}
