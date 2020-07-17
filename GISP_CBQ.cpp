/*_________________________________________________________________________________________________________
|                                                                                                          |
|                    CBQ algorithm for the Generalized Independent Set Problem (GISP)                      |
|                                                                                                          |
|                Copyright (c) 2018 Seyedmohammadhossein Hosseinian. All rights reserved.                  |
|                                                                                                          |
|__________________________________________________________________________________________________________|


 ***  READ ME  ********************************************************************************************

  (1) This code uses Intel(R) Math Kernel Library (MKL), an optimized version of LAPACK/BLAS libraries for 
      implementation on Intel(R) CPUs.
  (3) Intel(R) MKL library works only with "Intel C++" or "Visual C++" compilers.

 ***********************************************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <Windows.h>
#include <stdlib.h>
#include <stdio.h>
#include "mkl_lapacke.h"
#include <ppl.h>
#include <math.h>
#include <string>
#include <sstream>

#define timeLimit 10800

using namespace std;

#pragma region "Time Record"

double get_wall_time() {
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)) {
		return 0;
	}
	if (!QueryPerformanceCounter(&time)) {
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
}

double get_cpu_time() {
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
		return
			(double)(d.dwLowDateTime |
			((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	else {
		return 0;
	}
}

#pragma endregion

#pragma region "Heuristic"

struct elm {									//keeps a "index-value" pairs, used for sorting "indices" based on "values"
	int n;
	double val;
};

bool ValueCmp(elm const & a, elm const & b)		//comparison method based on "value" attribute of a "index-value" pair (i.e. elm)
{
	return a.val > b.val;
}

struct clq {									//clique: keeps list of vertices as well as the clique's total edge weight
	vector<int> vertexList;
	double weight;
};

double* makeQ(double * verW, double **Adj, int const & N) {					//generates the upper-triangle part of matrix Q, and puts it in a 1-d array to be used in the eigen-decomposition algorithm
	double** Q = new double*[N];
	for (int i = 0; i < N; i++) {
		Q[i] = new double[N];
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (j < i) {
				Q[i][j] = 0;
			}
			else {
				Q[i][j] = Adj[i][j];
			}
		}
	}
	for (int i = 0; i < N; i++) {
		Q[i][i] = 2 * verW[i];
	}
	double* Q_a = new double[N*N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			Q_a[i*N + j] = Q[i][j];
		}
	}
	for (int i = 0; i < N; i++)  delete[] Q[i];
	delete[] Q;
	return Q_a;
}

clq extractClique(double * verW, double **Adj_fix, double** Adj, vector<elm> & EVec, int const & N) {		//extracts clique based on a sorted list of "index-value" pair (will be used to extract clique based on a sorted eigenvector)
	clq clique;
	clique.vertexList.push_back(EVec[0].n);
	clique.weight = verW[EVec[0].n];
	for (int k = 1; k < N; k++) {
		bool belongs = true;
		double tempW = verW[EVec[k].n];
		for (int l = 0; l < clique.vertexList.size(); l++) {
			if (Adj_fix[EVec[k].n][clique.vertexList[l]] > 0) {
				tempW += Adj[EVec[k].n][clique.vertexList[l]];
				int check = (verW[EVec[k].n] < verW[clique.vertexList[l]] ? verW[clique.vertexList[l]] : verW[EVec[k].n]);
				if (-Adj[EVec[k].n][clique.vertexList[l]] > check) cout << "Error!" << endl;
			}
			else {
				belongs = false;
				break;
			}
		}
		if (belongs == true && tempW > 0) {
			clique.vertexList.push_back(EVec[k].n);
			clique.weight += tempW;
		}
	}
	return clique;
}

double CCH(double * verW, double **Adj_fix, double **Adj, int const & N)		// The base heuristic method
{
	double* Q_a = makeQ(verW, Adj, N);
	double* lambda = new double[N];
	double BestWeight = 0;
	clq tempClique;
	vector<elm> EVec(N);
	double** E = new double*[N];
	for (int i = 0; i < N; i++) {
		E[i] = new double[N];
	}
	MKL_INT n = N, lda = N, info;
	info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, Q_a, lda, lambda);
	if (info > 0) {
		printf("The algorithm failed to compute eigenvalues.\n");
		exit(1);
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			E[i][j] = Q_a[i*N + j];
		}
	}
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			EVec[i].n = i;
			EVec[i].val = E[i][j];
		}
		sort(EVec.begin(), EVec.end(), ValueCmp);
		tempClique = extractClique(verW, Adj_fix, Adj, EVec, N);
		if (tempClique.weight > BestWeight) {
			BestWeight = tempClique.weight;
		}
		for (int i = 0; i < N; i++) {
			EVec[i].val *= -1;
		}
		sort(EVec.begin(), EVec.end(), ValueCmp);
		tempClique = extractClique(verW, Adj_fix, Adj, EVec, N);
		if (tempClique.weight > BestWeight) {
			BestWeight = tempClique.weight;
		}
	}
	delete[] Q_a;
	delete[] lambda;
	for (int i = 0; i < N; i++)  delete[] E[i];
	delete[] E;
	return BestWeight;
}

double* makeQ_DP(double* verNeiW, double **vAdj, int vN) {
	double** vQ = new double*[vN];
	for (int i = 0; i < vN; i++) {
		vQ[i] = new double[vN];
	}
	for (int i = 0; i < vN; i++) {
		for (int j = 0; j < vN; j++) {
			if (j < i) {
				vQ[i][j] = 0;
			}
			else {
				vQ[i][j] = vAdj[i][j];
			}
		}
	}
	for (int i = 0; i < vN; i++) {
		vQ[i][i] = 2 * verNeiW[i];
	}
	double* Q_a = new double[vN*vN];
	for (int i = 0; i < vN; i++) {
		for (int j = 0; j < vN; j++) {
			Q_a[i*vN + j] = vQ[i][j];
		}
	}
	delete[] vQ;
	return Q_a;
}

clq extractClique_DP(double* verNeiW, double** vAdj_fix, double** vAdj, vector<elm> & EVec, int vN) {
	clq clique;
	clique.vertexList.push_back(EVec[0].n);
	clique.weight = verNeiW[EVec[0].n];
	for (int k = 1; k < vN; k++) {
		bool belongs = true;
		double tempW = verNeiW[EVec[k].n];
		for (int l = 0; l < clique.vertexList.size(); l++) {
			if (vAdj_fix[EVec[k].n][clique.vertexList[l]] > 0) {
				tempW += vAdj[EVec[k].n][clique.vertexList[l]];
				int check = (verNeiW[EVec[k].n] < verNeiW[clique.vertexList[l]] ? verNeiW[clique.vertexList[l]] : verNeiW[EVec[k].n]);
				if (-vAdj[EVec[k].n][clique.vertexList[l]] > check) cout << "Error!" << endl;
			}
			else {
				belongs = false;
				break;
			}
		}
		if (belongs == true && tempW > 0) {
			clique.vertexList.push_back(EVec[k].n);
			clique.weight += tempW;
		}
	}
	return clique;
}

double CCH_DP(double * verW, double **Adj_fix, double **Adj, int const & N)		// the CCH heuristic method in a dynamic programming setting
{
	double BestWeight = 0;
	Concurrency::parallel_for(0, N, [&](int v) {
		vector<int> vNeigh(1);
		vector<double> vNeighWei(1);
		vNeigh[0] = v;
		vNeighWei[0] = verW[v];
		for (int j = 0; j < N; j++) {
			if (Adj_fix[v][j] > 0) {
				vNeigh.push_back(j);
				vNeighWei.push_back(verW[j]);
			}
		}
		int vN = vNeigh.size();
		double** vAdj = new double*[vN];
		for (int i = 0; i < vN; i++) {
			vAdj[i] = new double[vN];
		}
		for (int i = 0; i < vN; i++) {
			for (int j = 0; j < vN; j++) {
				vAdj[i][j] = Adj[vNeigh[i]][vNeigh[j]];
			}
		}
		double** vAdj_fix = new double*[vN];
		for (int i = 0; i < vN; i++) {
			vAdj_fix[i] = new double[vN];
		}
		for (int i = 0; i < vN; i++) {
			for (int j = 0; j < vN; j++) {
				vAdj_fix[i][j] = Adj_fix[vNeigh[i]][vNeigh[j]];
			}
		}
		double* verNeiW = new double[vN];
		for (int i = 0; i < vN; i++) {
			verNeiW[i] = vNeighWei[i];
		}
		double* Q_a = makeQ_DP(verNeiW, vAdj, vN);
		double* lambda = new double[vN];
		clq tempClique;
		vector<elm> EVec(vN);
		double** E = new double*[vN];
		for (int i = 0; i < vN; i++) {
			E[i] = new double[vN];
		}
		MKL_INT n = vN, lda = vN, info;
		info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, Q_a, lda, lambda);
		if (info > 0) {
			printf("The algorithm failed to compute eigenvalues.\n");
			exit(1);
		}
		for (int i = 0; i < vN; i++) {
			for (int j = 0; j < vN; j++) {
				E[i][j] = Q_a[i*vN + j];
			}
		}
		for (int j = 0; j < vN; j++) {
			for (int i = 0; i < vN; i++) {
				EVec[i].n = i;
				EVec[i].val = E[i][j];
			}
			sort(EVec.begin(), EVec.end(), ValueCmp);
			tempClique = extractClique_DP(verNeiW, vAdj_fix, vAdj, EVec, vN);
			if (tempClique.weight > BestWeight) {
				BestWeight = tempClique.weight;
			}
			for (int i = 0; i < vN; i++) {
				EVec[i].val *= -1;
			}
			sort(EVec.begin(), EVec.end(), ValueCmp);
			tempClique = extractClique_DP(verNeiW, vAdj_fix, vAdj, EVec, vN);
			if (tempClique.weight > BestWeight) {
				BestWeight = tempClique.weight;
			}
		}
		for (int i = 0; i < vN; i++)  delete[] vAdj[i];
		delete[] vAdj;
		for (int i = 0; i < vN; i++)  delete[] vAdj_fix[i];
		delete[] vAdj_fix;
		delete[] verNeiW;
		delete[] Q_a;
		delete[] lambda;
		for (int i = 0; i < vN; i++)  delete[] E[i];
		delete[] E;
	});
	return BestWeight;
}

void print_matrix(char* desc, MKL_INT m, MKL_INT n, double* a, MKL_INT lda) {	//printing a eigenvalues and eigenvectors (to verify if necessary)
	MKL_INT i, j;
	printf("\n %s\n", desc);
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) printf(" %6.2f", a[i*lda + j]);
		printf("\n");
	}
}

#pragma endregion

#pragma region "Initial Sort"

struct node {
	int n;
	int degree;			//degree of the vertex in the subgraph induced by R (changing as R is updated)
	int ex_deg;			//sum of "degree" of the vertices adjacent to this vertex in the subgraph induced by R (See definition of ex_deg(q) in Tomita(2007) page 101)
};

bool degCmp(node const & a, node const & b)
{
	return a.degree > b.degree;
}

bool ex_degCmp(node const & a, node const & b)
{
	return a.ex_deg < b.ex_deg;
}

int* sortV(double** Adj, int & Delta, int const & N) {			//sorts the vertices based on "degree" and "ex_degree" (See definition of ex_deg(q) in Tomita(2007) page 101)
	int* V = new int[N];
	vector<node> R;
	vector<node> Rmin;
	node v;
	int dlt = 0;
	for (int i = 0; i < N; i++) {
		v.n = i;
		v.degree = 0;
		for (int j = 0; j < N; j++) {
			if (Adj[i][j] > 0) {
				v.degree += 1;
			}
		}
		if (v.degree > dlt) {
			dlt = v.degree;
		}
		R.push_back(v);
	}
	Delta = dlt;								//inputs Delta and change its value in the function after calculating the degree of all vertices
	sort(R.begin(), R.end(), degCmp);			//Sorts "node"s in R in a decreasing order "degree"
	int minDeg = (R.end() - 1)->degree;
	vector<node>::iterator itr = R.end() - 1;
	while (itr->degree == minDeg) {
		Rmin.push_back(*itr);
		if (itr == R.begin()) {
			break;
		}
		else {
			itr--;
		}
	}
	node p;										//The "node" with the minimum "ex_deg" among nodes in Rmin
	for (int k = N - 1; k >= 0; k--) {
		if (Rmin.size() == 1) {
			p = Rmin[0];
		}
		else {
			for (vector<node>::iterator itr_1 = Rmin.begin(); itr_1 != Rmin.end(); itr_1++) {
				itr_1->ex_deg = 0;
				for (vector<node>::iterator itr_2 = R.begin(); itr_2 != R.end(); itr_2++) {
					if (Adj[itr_1->n][itr_2->n] > 0) {
						itr_1->ex_deg += itr_2->degree;
					}
				}
			}
			sort(Rmin.begin(), Rmin.end(), ex_degCmp);				//Sorts "node"s in Rmin in an increasing order "ex_deg"
			p = Rmin[0];
		}
		V[k] = p.n;
		Rmin.clear();
		vector<node>::iterator itr = R.end() - 1;
		while (itr != R.begin()) {
			if (itr->n == p.n) {
				itr = R.erase(itr);
				break;
			}
			else {
				itr--;
			}
		}
		for (vector<node>::iterator itr_1 = R.begin(); itr_1 != R.end(); itr_1++) {
			if (Adj[itr_1->n][p.n] > 0) {
				itr_1->degree -= 1;
			}
		}
		sort(R.begin(), R.end(), degCmp);
		minDeg = (R.end() - 1)->degree;
		itr = R.end() - 1;
		while (itr->degree == minDeg) {
			Rmin.push_back(*itr);
			if (itr == R.begin()) {
				break;
			}
			else {
				itr--;
			}
		}
	}
	return V;
}

#pragma endregion

#pragma region "BnB Search"

struct vertex {
	int n;
};

class Equation {
public:										//lam, b_p, q_p are parameters of the eqution
	vector<double> lam;
	vector<double> b_p;
	vector<double> q_p;
	double m;
public:
	Equation(vector<double> _lambda, vector<double> _b_p, vector<double> _q_p, double _m) :lam(_lambda), b_p(_b_p), q_p(_q_p), m(_m) {}

	double evalFunc(double x) {				//evaluates function value at point x
		double f = -m / 4;
		double num;
		double denum;
		for (int i = 0; i < m; i++) {
			num = ((lam[i] * b_p[i]) + q_p[i])*((lam[i] * b_p[i]) + q_p[i]);
			denum = (lam[i] - x)*(lam[i] - x);
			f += (num / denum);
		}
		return f;
	}

	double root(double a, double b) {		//finds the root of a monotonic function in the interval [a,b]
		double c = a;
		double fa = evalFunc(a);
		double fb = evalFunc(b);
		double fc = fa;
		do {
			double d1 = b - a;
			if (fabs(fc) < fabs(fb)) {
				a = b; b = c; c = a;
				fa = fb; fb = fc; fc = fa;
			}
			double d2 = (c - b) / 2.0;
			double eps_one = DBL_EPSILON*(2.0*fabs(b) + 0.5);
			double eps = (eps_one < 0.0001 ? eps_one : 0.0001);
			if (fabs(d2) <= eps || !fb) return b;
			if (fabs(d1) >= eps&&fabs(fa) > fabs(fb)) {
				double p, q;
				double cb = c - b;
				double t1 = fb / fa;
				if (a == c) {
					p = cb*t1;
					q = 1.0 - t1;
				}
				else {
					double t2 = fb / fc;
					q = fa / fc;
					p = t1*(cb*q*(q - t2) - (b - a)*(t2 - 1.0));
					q = (q - 1.0)*(t1 - 1.0)*(t2 - 1.0);
				}
				if (p > 0.0)q = -q;
				else p = -p;
				if (2.0*p < 1.5*cb*q - fabs(eps*q) && 2.0*p < fabs(d1))d2 = p / q;
			}
			if (fabs(d2) < eps)d2 = (d2 > 0.0 ? eps : -eps);
			a = b;
			fa = fb;
			b += d2;
			fb = evalFunc(b);
			if (fb > 0.0&&fc > 0.0 || fb < 0.0&&fc < 0.0) {
				c = a;
				fc = fa;
			}
		} while (true);
	}
};

double* makeQ_L(double** adj, int m) {			//generates the upper_triangle part of the matrix Q_L, and puts it a 1-d array to be used in the eigen-decomposition algorithm
	double** Q_L = new double*[m];
	for (int i = 0; i < m; i++) {
		Q_L[i] = new double[m];
	}
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			if (j < i) {
				Q_L[i][j] = 0;
			}
			else {
				Q_L[i][j] = adj[i][j];
			}
		}
	}
	double* Q_La = new double[m*m];
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			Q_La[i*m + j] = Q_L[i][j];
		}
	}
	for (int i = 0; i < m; i++)  delete[] Q_L[i];
	delete[] Q_L;
	return Q_La;
}

bool prune(double * verW, double ** Adj, const vector<vertex> & candList, const vector<int> & S, double W, double Wmax, int & qp_bound_count, int & trv_bound_count, double & qp_trv, int & badMAT) {
	bool prn = false;
	int d = S.size();
	int m = candList.size();
	vector<double> b(m, 0.5);
	vector<double> q(m, 0);
	double candPureVerWeight = 0;
	for (int i = 0; i < m; i++) {
		q[i] = verW[candList[i].n];
		candPureVerWeight += q[i];
	}
	if (d != 0) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < d; j++) {
				q[i] += Adj[candList[i].n][S[j]];
			}
		}
	}
	double Z_L = 0;
	if (m == 1) {
		Z_L += q[0];
	}
	else {
		double** adj_L = new double*[m];
		for (int i = 0; i < m; i++) {
			adj_L[i] = new double[m];
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				adj_L[i][j] = Adj[candList[i].n][candList[j].n];
			}
		}
		double* Q_La = makeQ_L(adj_L, m);
		double* lambda = new double[m];
		double** E = new double*[m];
		for (int i = 0; i < m; i++) {
			E[i] = new double[m];
		}
		MKL_INT n = m, lda = m, info;
		info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, Q_La, lda, lambda);
		if (info > 0) {
			printf("The algorithm failed to compute eigenvalues.\n");
			exit(1);
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < m; j++) {
				E[i][j] = Q_La[i*m + j];
			}
		}
		vector<double> _b_p(m, 0);
		vector<double> _q_p(m, 0);
		for (int j = 0; j < m; j++) {
			for (int i = 0; i < m; i++) {
				_b_p[j] += E[i][j] * b[i];
				_q_p[j] += E[i][j] * q[i];
			}
		}
		vector<double> _lambda(m);
		for (int i = 0; i < m; i++) {
			_lambda[i] = lambda[i];
		}
		double lambda_max = *(_lambda.end() - 1);
		int k = m - 1;
		while (lambda[k] == lambda_max) {
			if (lambda[k] * _b_p[k] + _q_p[k] != 0) {
				Equation eqa(_lambda, _b_p, _q_p, m);
				double lower_bound = *(_lambda.end() - 1) + 1e-7;
				double upper_bound = 2 * lower_bound + 10.0;
				while (eqa.evalFunc(upper_bound) > 0.0) {
					lower_bound = upper_bound;
					upper_bound *= 2;
				}
				double mu = eqa.root(lower_bound, upper_bound);
				double num_1;
				double num_2;
				double denum;
				for (int i = 0; i < m; i++) {
					num_1 = (mu * _b_p[i]) + _q_p[i];
					num_2 = (_lambda[i] * ((mu * _b_p[i]) - _q_p[i])) + (2 * mu * _q_p[i]);
					denum = (_lambda[i] - mu) * (_lambda[i] - mu);
					Z_L += ((num_1 * num_2) / denum);
				}
				Z_L = 0.5 * Z_L;
				break;
			}
			k--;
		}
		for (int i = 0; i < m; i++)  delete[] adj_L[i];
		delete[] adj_L;
		delete[] Q_La;
		delete[] lambda;
		for (int i = 0; i < m; i++)  delete[] E[i];
		delete[] E;
	}
	double upperBound;
	if (Z_L == 0) {
		badMAT++;
		upperBound = candPureVerWeight;
	}
	else if (Z_L <= candPureVerWeight) {
		qp_bound_count++;
		upperBound = Z_L;
	}
	else {
		trv_bound_count++;
		upperBound = candPureVerWeight;
	}
	qp_trv += Z_L / candPureVerWeight;
	if (W + upperBound <= Wmax) {
		prn = true;
	}
	return prn;
}

void EXPAND(double * verW, double ** Adj_fix, double ** Adj, vector<vertex> & U, vector<int> & S, double & W, double & Wmax, int & count, int & qp_bound_count, int & trv_bound_count, double & qp_trv, int & badMAT, double & cpu_2, bool & checkTime) {
	count++;
	while (!U.empty()) {
		vertex v;
		v.n = (U.end() - 1)->n;
		bool prn = prune(verW, Adj, U, S, W, Wmax, qp_bound_count, trv_bound_count, qp_trv, badMAT);
		if (prn == false) {
			double addedWeight = verW[v.n];
			if (S.size() != 0) {
				for (vector<int>::iterator Sit = S.begin(); Sit != S.end(); Sit++) {
					addedWeight += Adj[*Sit][v.n];
				}
			}
			W += addedWeight;
			S.push_back(v.n);
			if (W > Wmax) Wmax = W;
			vector<vertex> Uv;
			for (vector<vertex>::iterator itr_1 = U.begin(); itr_1 != U.end(); itr_1++) {
				if (Adj_fix[itr_1->n][v.n] > 0) {
					Uv.push_back(*itr_1);
				}
			}
			if (!Uv.empty()) {
				EXPAND(verW, Adj_fix, Adj, Uv, S, W, Wmax, count, qp_bound_count, trv_bound_count, qp_trv, badMAT, cpu_2, checkTime);
				double timeLapsed = get_cpu_time() - cpu_2;
				if (timeLapsed > double(timeLimit)) {
					checkTime = true;
					return;
				}
			}
			vector<int>::iterator itr = S.erase(S.end() - 1);
			W -= addedWeight;
		}
		else {
			return;
		}
		vector<vertex>::iterator Uit;
		if (U.size() == 1) {
			U.clear();
		}
		else {
			Uit = U.end() - 1;
			while (Uit != U.begin()) {
				if (Uit->n == v.n) {
					Uit = U.erase(Uit);
					break;
				}
				else {
					Uit -= 1;
				}
			}
		}
	}
	return;
}
#pragma endregion

int main()
{
	cout << endl;
	cout << "=====================================================" << endl;
	cout << "          CBQ Algorithm for the GIS problem          " << endl;
	cout << "=====================================================" << endl;
	cout << endl;
	filebuf fb;
	fb.open("#output.txt", ios::out);
	ostream os(&fb);
	int verCardin;
	string GRAPH;
	string line;
	string initial_word, word_1, word_2, word_3;
	int s, t, weigh;
	ifstream instance("#instance_GISP.txt");
	while (instance >> verCardin >> GRAPH) {
		int m_1 = 0;
		int m_2 = 0;
		int ver_num = 0;
		int N = verCardin;
		double* verW = new double[N];
		for (int i = 0; i < N; i++) {
			verW[i] = 0;
		}
		double** Adj = new double*[N];
		for (int i = 0; i < N; i++) {
			Adj[i] = new double[N];
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				Adj[i][j] = 0;
			}
		}
		double** Adj_fix = new double*[N];
		for (int i = 0; i < N; i++) {
			Adj_fix[i] = new double[N];
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				Adj_fix[i][j] = 0;
			}
		}
		double** Adj_rem = new double*[N];
		for (int i = 0; i < N; i++) {
			Adj_rem[i] = new double[N];
		}
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				Adj_rem[i][j] = 0;
			}
		}
		ifstream data(GRAPH);
		while (getline(data, line)) {
			istringstream iss(line);
			iss >> initial_word;
			if (initial_word == "e") {
				m_1++;
				iss >> word_1 >> word_2;
				stringstream make_num_1(word_1);
				make_num_1 >> s;
				stringstream make_num_2(word_2);
				make_num_2 >> t;
				Adj[s - 1][t - 1] = 1;
				Adj[t - 1][s - 1] = 1;
				Adj_fix[s - 1][t - 1] = 1;
				Adj_fix[t - 1][s - 1] = 1;
			}
			else if (initial_word == "n") {
				ver_num++;
				iss >> word_1 >> word_2;
				stringstream make_num_1(word_1);
				make_num_1 >> s;
				stringstream make_num_2(word_2);
				make_num_2 >> weigh;
				verW[s - 1] = weigh;
			}
			else if (initial_word == "not_e") {
				m_2++;
				iss >> word_1 >> word_2 >> word_3;
				stringstream make_num_1(word_1);
				make_num_1 >> s;
				stringstream make_num_2(word_2);
				make_num_2 >> t;
				stringstream make_num_3(word_3);
				make_num_3 >> weigh;
				Adj[s - 1][t - 1] = 1;
				Adj[t - 1][s - 1] = 1;
				Adj_rem[s - 1][t - 1] = weigh;
				Adj_rem[t - 1][s - 1] = weigh;
			}
		}
		for (int i = 0; i < N; i++) {							//Make Adj the "weighted" adjacency matrix of the graph
			for (int j = 0; j < N; j++) {
				if (Adj[i][j] != 0) {
					if (Adj_fix[i][j] == 1) {
						Adj[i][j] = (verW[i] < verW[j] ? -verW[i] - 1 : -verW[j] - 1);
					}
					else {
						Adj[i][j] = -1 * Adj_rem[i][j];
					}
				}
			}
		}
		for (int i = 0; i < N; i++) {							//Hereafter, Adj_fix will be the adjacency matrix of the complement of G1=(V,E1), where E1 is the set of fixed edges
			for (int j = 0; j < N; j++) {
				if (i != j) {
					if (Adj_fix[i][j] == 1) Adj_fix[i][j] = 0;
					else Adj_fix[i][j] = 1;
				}
				else {
					Adj_fix[i][j] = 0;
				}
			}
		}
		bool checkTime = false;
		int count = 0;
		int qp_bound_count = 0;
		int trv_bound_count = 0;
		double qp_trv = 0.0;
		int badMAT = 0;
		vector<int> S;
		double W = 0;
		double cpu_1 = get_cpu_time();
		/**************************************************/
		double InitialSol = 0;
		//double InitialSol = CCH(verW, Adj_fix, Adj, N);
		//double InitialSol = CCH_DP(verW, Adj_fix, Adj, N);
		/**************************************************/
		double cpu_2 = get_cpu_time();
		double Wmax = InitialSol;
		int Delta;
		int* V = sortV(Adj_fix, Delta, N);
		vector<vertex> U;
		vertex u;
		for (int i = 0; i < N; i++) {
			u.n = V[i];
			U.push_back(u);
		}
		/*******************************************************************************************************************/
		EXPAND(verW, Adj_fix, Adj, U, S, W, Wmax, count, qp_bound_count, trv_bound_count, qp_trv, badMAT, cpu_2, checkTime);
		/*******************************************************************************************************************/
		double cpu_3 = get_cpu_time();
		os << endl;
		os << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << endl;
		os << "XXXXXX    CBQ Algorithm for the GIS problem    XXXXXX" << endl;
		os << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << endl;
		os << endl;
		os << "INSTANCE :              " << GRAPH << endl;
		os << "# Vertices :            " << ver_num << endl;
		os << "# Fixed edges :         " << m_1 << endl;
		os << "# Removable edges :     " << m_2 << endl;
		int mm_1 = 0;
		int mm_2 = 0;
		for (int i = 0; i < N - 1; i++) {
			for (int j = i + 1; j < N; j++) {
				if (Adj_fix[i][j] == 0) mm_1++;
				if (Adj_rem[i][j] > 0) mm_2++;
			}
		}
		cout << "INSTANCE :              " << GRAPH << endl;
		cout << "# Vertices :            " << ver_num << endl;
		if (m_1 == mm_1) cout << "# Fixed edges :         " << m_1 << endl;
		else cout << "ERROR in number of Fixed edges!" << endl;
		if (m_2 == mm_2) cout << "# Removable edges :     " << m_2 << endl;
		else cout << "ERROR in number of Removable edges!" << endl;
		os << endl;
		if (checkTime == true) {
			os << "   ******************************************** " << endl;
			os << "   ***   Terminated due to the TIME limit   ***   " << endl;
			os << "   ******************************************** " << endl << endl;
		}
		os << "Heuristic Sol. :        " << InitialSol << endl;
		cout << "Heuristic Sol. :        " << InitialSol << endl;
		os << "# of BnB nodes :        " << count << endl;
		os << "Max Weight :            " << Wmax << endl;
		cout << "Max Weight :            " << Wmax << endl;
		os << "-----------------------------------------------------" << endl;
		os << "Heur. CPU time (ms)  :  " << 1000 * (cpu_2 - cpu_1) << endl;
		os << "B&B CPU time (ms)  :    " << 1000 * (cpu_3 - cpu_2) << endl;
		os << "Total CPU time (ms)  :  " << 1000 * (cpu_3 - cpu_1) << endl;
		os << endl;
		os << "=====================================================" << endl;
		cout << "=====================================================" << endl;
		for (int i = 0; i < N; i++)  delete[] Adj_rem[i];
		delete[] Adj_rem;
		for (int i = 0; i < N; i++)  delete[] Adj_fix[i];
		delete[] Adj_fix;
		for (int i = 0; i < N; i++)  delete[] Adj[i];
		delete[] Adj;
		delete[] verW;
	}
	fb.close();
};
