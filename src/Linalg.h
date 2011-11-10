//Templated C-Style vector operations

#ifndef LINALG_H
#define LINALG_H

#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <limits>

#include "Matrix.h"
#include "DenseMatrix.h"
#include "Vector.h"
#include "DenseVector.h"


#define isDoubleTPrecision() sizeof(TPrecision) == sizeof(double)

//Checks bounds for matrix multuiplications, etc.
//make it slightly slower
#define LINALG_CHECK

//blas routines
namespace blas{

  extern "C"{

  //---- matrix matrix multiply ----//
  
  void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
      double *alpha, double *A, int *lda, double *B, int *ldb, double *beta,
      double *C, int *ldc);
    
  void sgemm_(char *transa, char *transb, int *m, int *n, int *k, 
      float *alpha, float *A, int *lda, float *B, int *ldb, float *beta, 
      float *C, int *ldc);
  



  //---- matrix vector multiply ----//
  
  void dgemv_(char *trans, int *m, int *n, double *alpha, double *A, int *lda, 
      double *x, int *incx, double *beta, double *y, int *incy);
    
  void sgemv_(char *trans, int *m, int *n, float *alpha, float *A, int *lda, 
      float *x, int *incx, float *beta, float *y, int *incy);





  //---- dot product ----/
  
  double ddot_(int *m, double *x, int *incx, double *y, int *incy);

  float sdot_(int *m, float *x, int *incx, float *y, int *incy);


  //---- QR decomposition ----//
  void dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
  void sgeqrf_(int *m, int *n, float *a, int *lda, float *tau, float *work, int *lwork, int *info);

  void dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
  void sorgqr_(int *m, int *n, int *k, float *a, int *lda, float *tau, float *work, int *lwork, int *info);

  //---- LU decvomposition ----//
  void dgetrf_(int * m, int *n, double *a, int *lda, int *ipiv, int *info);

  void sgetrf_(int * m, int *n, float *a, int *lda, int *ipiv, int *info);
  
  //---- Cholesky decvomposition ----//
  void dpotrf_(char *u, int *n, double *a, int *lda, int *info);

  void spotrf_(char *u, int *n, float *a, int *lda, int *info);
  
  //---- inverse ---//
  void dgetri_(int *n, double *a, int *lda, int *ipiv, double *work, int
      *lwork, int *info);

  void sgetri_(int *n, float *a, int *lda, int *ipiv, float *work, int
      *lwork, int *info);


  //---- inverse symmetric positive definite matrix ----/
  
  void dpotri_(char *uplo, int *n, double *A, int *lda, int *info);
  
  void spotri_(char *uplo, int *n, float *A, int *lda, int *info);

  //---- Solve SPD linear equations ----//

  void dposv_(char *uplo, int *n, int *nrhs, double *A, int *lda, double *B,
      int *ldb, int *info );

  void sposv_(char *uplo, int *n, int *nrhs, float *A, int *lda, float *B,
      int *ldb, int *info );

  void dposvx_( char *fact, char *uplo, int *n, int *nrhs, double *A, int *lda,
      double *af, int *ldaf, char *equed, double *s, double *b, int *ldb, double
      *x, int *ldx, double *rcond, double *ferr, double *berr, double *work,
      int *iwork, int *info );
  
  void sposvx_( char *fact, char *uplo, int *n, int *nrhs, float *A, int *lda,
      float *af, int *ldaf, char *equed, float *s, float *b, int *ldb, float
      *x, int *ldx, float *rcond, float *ferr, float *berr, float *work,
      int *iwork, int *info );

  //---- Solve linear equations ----/
  
  void dgesv_( int *n, int *nrhs, double *A, int *lda, int *ipiv, double *b, int
      *ldb, int *info );

  void sgesv_( int *n, int *nrhs, float *A, int *lda, int *ipiv, float *b, int
      *ldb, int *info );

  //---- Least squares  or minimum norm with QR or LQ, A is assumed to have full rank----// 
  void dgels_(char *trans, int *m, int *n, int *nhrs, double *A, int *lda, double
      *B, int *ldb, double *work, int *lwork, int *info);

  void sgels_(char *trans, int *m, int *n, int *nhrs, float *A, int *lda, float 
      *B, int *ldb, float *work, int *lwork, int *info);

  //---- Least squares  or minimum norm with SVD// 
  void dgelsd_(int *m, int *n, int *nhrs, double *A, int *lda, double
      *B, int *ldb, double *S, double *rcond, int *rank, double *work, int
      *lwork, int *iwork, int *info);

  void sgelsd_(int *m, int *n, int *nhrs, float *A, int *lda, float
      *B, int *ldb, float *S, float *rcond, int *rank, float *work, int *lwork,
      int *iwork, int *info);


  //--- Cholesky condition number ---//
  void dpocon(char *uplo, int *n, double *a, int *lda, double *anorm, double
      *rcond, double *work, int *iwork, int *info);
  
  void spocon(char *uplo, int *n, float *a, int *lda, float *anorm, float
      *rcond, float *work, int *iwork, int *info);

  };
};


template<typename TPrecision>
class Linalg{


  public:

  
  //--------- Methods for DenseMatrices/Vectors using blas - fast -----------//

  
  //least squares with svd
  static DenseMatrix<TPrecision> LeastSquares(DenseMatrix<TPrecision> &a,
      DenseMatrix<TPrecision> &b, double *sse = NULL){


    int m = a.M();
    int n = a.N();
    int lda = m;

    int nrhs = b.N();
    int ldb = std::max(m, n);
    int info = 0;
    TPrecision workTmp = 0 ;
    int query = -1;
    bool deallocate = false; 
    DenseMatrix<TPrecision> tmp = b; 
    if(ldb > (int) b.M()){
      deallocate = true;
      tmp = DenseMatrix<TPrecision>(ldb, nrhs);
      for(unsigned int i=0; i<b.M(); i++){
        for(unsigned int j=0; j<b.N(); j++){
          tmp(i, j)= b(i, j);
        }
      }
    }

    TPrecision *s = new TPrecision[std::min(m, n)];
    TPrecision rcond = -1; //use machine precision as condition number
    int rank = -1;
    int *iwork = new int[100*std::min(m,n)];
    
    if(isDoubleTPrecision()){
      blas::dgelsd_(&m, &n, &nrhs, (double*)a.data(), &lda,
          (double*)tmp.data(), &ldb,  (double*)s, (double*)&rcond, &rank,
          (double*)&workTmp, &query, iwork, &info);

      int lwork = workTmp;
      double *work =new double[lwork];
      blas::dgelsd_(&m, &n, &nrhs, (double*)a.data(), &lda,
          (double*)tmp.data(), &ldb,  (double*)s, (double*)&rcond, &rank,
          (double*)work, &lwork, iwork, &info);
      delete[] work;
    }
    else{
      blas::sgelsd_(&m, &n, &nrhs, (float*)a.data(), &lda,
          (float*)tmp.data(), &ldb,  (float*)s, (float*)&rcond, &rank,
          (float*)&workTmp, &query, iwork, &info);

      int lwork = workTmp;
      float *work =new float[lwork];
 
      blas::sgelsd_(&m, &n, &nrhs, (float*)a.data(), &lda,
          (float*)tmp.data(), &ldb,  (float*)s, (float*)&rcond, &rank,
          (float*)work, &lwork, iwork, &info);
      delete[] work;
    }

    int nrows = a.N();
    DenseMatrix<TPrecision> out(nrows, nrhs);
    for(unsigned int i=0; i<out.M(); i++){
      for(unsigned int j=0; j<out.N(); j++){
        out(i, j) = tmp(i, j);
      }
    }

    if(sse != NULL){
      for(unsigned int i=0; i<out.N(); i++){
        sse[i] = 0;
        for(unsigned int j=out.M(); j < tmp.M(); j++){
          double blah = tmp(j, i);
          sse[i] += blah *  blah;
        }
      }
    }
    
    if(deallocate){
      tmp.deallocate();
    }


    //std::cout << "Info " << info << "  Rank: " << rank << std::endl;
    delete[] s;
    delete[] iwork;

    return out;

  };



  //--- linear equations

  static DenseMatrix<TPrecision> Solve(DenseMatrix<TPrecision> &a, DenseMatrix<TPrecision> &b){
        DenseMatrix<TPrecision> x(b.M(), b.N());
        Copy(b, x);
        Solve2(a, x);
        return x;
  };

  static bool Solve2(DenseMatrix<TPrecision> &a, DenseMatrix<TPrecision> &b){

    int n = a.M();
    int nrhs = b.N();
    int lda = n;
    int ldb = b.M();
    int info = 0;
    int *ipiv = new int[n];
    

    if(isDoubleTPrecision()){
      blas::dgesv_(&n, &nrhs, (double*)a.data(), &lda, ipiv, (double*)b.data(), &ldb, &info);
    }
    else{ 
      blas::sgesv_(&n, &nrhs, (float*)a.data(), &lda, ipiv, (float*)b.data(), &ldb, &info);
    }

    delete[] ipiv;

    return info == 0;

  };

  //--- SPD linear equations

  static DenseMatrix<TPrecision> SolveSPD(DenseMatrix<TPrecision> &a, DenseMatrix<TPrecision> &b){
        DenseMatrix<TPrecision> x(b.M(), b.N());
        Copy(b, x);
        SolveSPD2(a, x);
        return x;
  };
  
  static DenseMatrix<TPrecision> SolveSPD(DenseMatrix<TPrecision> &a,
      DenseMatrix<TPrecision> &b, TPrecision &rcond, int &info){
        DenseMatrix<TPrecision> x(b.M(), b.N());
        Copy(b, x);
        SolveSPD2(a, x, rcond, info);
        return x;
  };

  
  static bool SolveSPD2(DenseMatrix<TPrecision> &a, DenseMatrix<TPrecision> &b){

    char u='U';
    int n = a.M();
    int nrhs = b.N();
    int lda = n;
    int ldb = b.M();
    int info = 0;
    

    if(isDoubleTPrecision()){
      blas::dposv_(&u, &n, &nrhs, (double*)a.data(), &lda, (double*)b.data(), &ldb, &info);
    }
    else{ 
      blas::sposv_(&u, &n, &nrhs, (float*)a.data(), &lda, (float*)b.data(), &ldb, &info);
    }

    return info == 0;

  };


  static void SolveSPD2(DenseMatrix<TPrecision> &a, DenseMatrix<TPrecision> &b,
      TPrecision &rcond, int &info){

    char fact ='E';
    char u='U';
    int n = a.M();
    int nrhs = b.N();
    int lda = n;
    TPrecision *af = new TPrecision[n*n];
    int ldaf = n;
    char equed ='0';
    TPrecision *s = new TPrecision[n];
    int ldb = b.M();
    TPrecision *x = new TPrecision[n * nrhs];
    int ldx = n;
    TPrecision *ferr = new TPrecision[nrhs];
    TPrecision *berr = new TPrecision[nrhs];
    TPrecision *work = new TPrecision[3*n];
    int *iwork = new int[n];


    if(isDoubleTPrecision()){
      blas::dposvx_(&fact, &u, &n, &nrhs, (double*)a.data(), &lda,
      (double*) af, &ldaf, &equed, (double*) s, (double*) b.data(), &ldb, 
      (double*) x, &ldx, (double*)&rcond, (double*)ferr, (double*)berr, (double*)work,
      iwork, &info );   
    }
    else{ 
      blas::sposvx_(&fact, &u, &n, &nrhs, (float*)a.data(), &lda,
      (float*) af, &ldaf, &equed, (float*) s, (float*) b.data(), &ldb, 
      (float*) x, &ldx, (float*)&rcond, (float*)ferr, (float*)berr, (float*)work,
      iwork, &info );   
    }

    delete[] iwork;
    delete[] x;
    delete[] s;
    delete[] work;
    delete[] berr;
    delete[] ferr;
    delete[] af;

  };

  //--- Condition numbers  ---//


  //--- Determinants ---//
  
  static TPrecision DetSPD(DenseMatrix<TPrecision> &a){
    DenseMatrix<TPrecision> ch = Cholesky(a);
    TPrecision d = DetSPDCholesky(ch);
    ch.deallocate();
    return d;
  };
  
  static TPrecision DetSPDCholesky(DenseMatrix<TPrecision> &ch){
     TPrecision d = 1;
    for(unsigned int i=0; i< ch.N(); i++){
      d *= ch(i, i);
    }
    return d*d;
  };


  static TPrecision Det(DenseMatrix<TPrecision> &a){
    DenseMatrix<TPrecision> lu = LU(a);
    TPrecision d = DetLU(lu);
    lu.deallocate();
    return d;
  };
  
  static TPrecision DetLU(DenseMatrix<TPrecision> &lu){
    TPrecision d = 1;
    for(unsigned int i=0; i< lu.N(); i++){
      d *= lu(i, i);
    }
    return d;
  };

  //---- LU factorization ---//
  static DenseMatrix<TPrecision> LU(DenseMatrix<TPrecision> &a, int *ipiv = NULL){
    int m = a.M();
    int n = a.N();
    DenseMatrix<TPrecision> lu = Copy(a);
    int lda = m;

    bool clear = false;
    if(ipiv == NULL){
     clear = true;
     ipiv = new int[std::min(n, m)];
    }
    int info = 0;
    
    if(isDoubleTPrecision()){
      blas::dgetrf_(&m, &n, (double*) lu.data(), &lda, ipiv, &info);
    }
    else{
      blas::sgetrf_(&m, &n, (float*) lu.data(), &lda, ipiv, &info);
    }

    if(clear){
      delete ipiv;
    }

    if(info !=0 ){
      //std::cerr << "LU error: " << info << std::endl;
    }
    return lu;
  };



  //---- QRfactorization ---//
  static DenseMatrix<TPrecision> QR(DenseMatrix<TPrecision> &a){
    DenseMatrix<TPrecision> q = Copy(a);
    Linalg<TPrecision>::QR_inplace(q);
    return q;    
  };

  static void QR_inplace(DenseMatrix<TPrecision> &q){
    int n = q.N();
    int m = q.M();
    int lda = m;
    int info = 0;
    TPrecision *work = new TPrecision[1];
    int lwork = -1;
    TPrecision *tau = new TPrecision[n];

    //workspace query
    if(isDoubleTPrecision()){
       blas::dgeqrf_(&m, &n, (double*) q.data(), &lda, (double*) tau, (double*) work, &lwork, &info);
    }
    else{ 
       blas::sgeqrf_(&m, &n, (float*) q.data(), &lda, (float*) tau, (float*) work, &lwork, &info);
    }
    if(info !=0 ){
      std::cerr << "QRF workspace error: " << info << std::endl;
    }

    lwork = work[0];
    delete work;
    work = new TPrecision[lwork];

    //qr
    if(isDoubleTPrecision()){
      blas::dgeqrf_(&m, &n, (double*) q.data(), &lda, (double*) tau, (double*) work, &lwork, &info);
      if(info !=0 ){
        std::cerr << "QRF error: " << info << std::endl;
      }

      blas::dorgqr_(&m, &n, &n, (double*) q.data(), &lda, (double*) tau, (double*) work, &lwork, &info);
      if(info !=0 ){
        std::cerr << "QR error: " << info << std::endl;
      }
    }
    else{
      blas::sgeqrf_(&m, &n, (float*) q.data(), &lda, (float*) tau, (float*) work, &lwork, &info);
      if(info !=0 ){
        std::cerr << "QRF error: " << info << std::endl;
      }

      blas::sorgqr_(&m, &n, &n, (float*) q.data(), &lda, (float*) tau, (float*) work, &lwork, &info);
      if(info !=0 ){
        std::cerr << "QR error: " << info << std::endl;
      }
    }


    delete work;
    delete tau;
  };



  //---- Choelsky factorization ---//
  static DenseMatrix<TPrecision> Cholesky(DenseMatrix<TPrecision> &a, char uplo = 'U'){
    int n = a.N();
    DenseMatrix<TPrecision> ch = Copy(a);
    int lda = n;
    int info = 0;
    
    if(isDoubleTPrecision()){
      blas::dpotrf_(&uplo, &n, (double*) ch.data(), &lda, &info);
    }
    else{ 
      blas::spotrf_(&uplo, &n, (float*) ch.data(), &lda, &info);
    }

    if(info !=0 ){
      //std::cerr << "Cholesky error: " << info << std::endl;
    }

    return ch;
  };



  //---- Inverse ---//
  static DenseMatrix<TPrecision> Inverse(DenseMatrix<TPrecision> &a){
    int *ipiv = new int[a.N()];
    DenseMatrix<TPrecision> inv = LU(a, ipiv);
    InverseLU(inv, ipiv);

    return inv;
  };

  //---- Inverse, inpout is lu factorizatiopn ---//
  static void InverseLU(DenseMatrix<TPrecision> &inv, int *ipiv){
    int n =inv.N();
    int lda = n;
    TPrecision *work = new TPrecision[2];
    int info = 0;
    int lwork = -1;

    if(isDoubleTPrecision()){
      blas::dgetri_(&n, (double*)inv.data(), &lda, ipiv,(double*) work, &lwork, &info);
    }
    else{
      blas::sgetri_(&n, (float*)inv.data(), &lda, ipiv, (float*) work, &lwork, &info);
    }

    lwork = work[0];
    delete work;
    work = new TPrecision[lwork];

    if(isDoubleTPrecision()){
      blas::dgetri_(&n, (double*)inv.data(), &lda, ipiv, (double*)work, &lwork, &info);
    }
    else{
      blas::sgetri_(&n, (float*)inv.data(), &lda, ipiv, (float*)work, &lwork, &info);
    }


    delete work;
    delete ipiv;
 
    if(info !=0 ){
      //std::cerr << "Inverse error: " << info << std::endl;
    }

  };

  //--- Symmetric positive definite inverse
  static DenseMatrix<TPrecision> InverseSPD(DenseMatrix<TPrecision> &a){
    char u ='U';
    DenseMatrix<TPrecision> inv = Cholesky(a, u);
    InverseCholesky(inv, u);
    return inv;
  }; 


  //--- Symmetric positive definite inverse
  static void InverseSPD(DenseMatrix<TPrecision> &a, DenseMatrix<TPrecision> &inv){
    char u ='U';
    Linalg<TPrecision>::Copy(a, inv);
    Cholesky(inv, u);
    InverseCholesky(inv, u);
  };   
  
  //--- Symmetric positive definite inverse, input is the choelsky decomposition
  static void InverseCholesky(DenseMatrix<TPrecision> &inv, char u = 'U'){

    int n =inv.N();
    int lda = n;
    int info = 0;
    if(isDoubleTPrecision()){
      blas::dpotri_(&u, &n, (double*)inv.data(), &lda, &info);
    }
    else{
      blas::spotri_(&u, &n, (float*)inv.data(), &lda, &info);
    }

    for(int i=0; i<n;i++){
      for(int j=i+1; j<n; j++){
        inv(j, i) = inv(i, j);
      }
    }
 
    if(info !=0 ){
      //std::cerr << "InverseSPD error: " << info << std::endl;
    }

  };
   

  //--- Matrix Matrix multiplication methods 
  
  //Matrix matrix multiply without output allocation
  static void Multiply(DenseMatrix<TPrecision> &a, DenseMatrix<TPrecision> &b,
      DenseMatrix<TPrecision> &c, bool transposeA = false, bool transposeB = false, 
      TPrecision alpha = 1 ){

    TPrecision beta = 0;

    char transa;
    char transb;
    int m;
    int n;
    int k;
    int lda;
    int ldb;

    if(transposeA){
      m = a.N();
      k = a.M();
      transa = 'T';

#ifdef LINALG_CHECK
      if(c.M() != a.N()) throw "Invalid matrix mult";
#endif

    }
    else{
      m = a.M();
      k = a.N();
      transa = 'N';

#ifdef LINALG_CHECK
      if(c.M() != a.M()) throw "Invalid matrix mult";
#endif

    }
    lda = a.M();

    if(transposeB){
      n = b.M();
      transb = 'T';

#ifdef LINALG_CHECK
      if(c.N() != b.M()) throw "Invalid matrix mult";
#endif

    }
    else{
      n = b.N();
      transb = 'N';

#ifdef LINALG_CHECK
      if(c.N() != b.N()) throw "Invalid matrix mult";
#endif
    }
    ldb = b.M();
    

    if(isDoubleTPrecision()){
      blas::dgemm_(&transa, &transb, &m, &n, &k, (double*)&alpha, (double*)a.data(),
          &lda, (double*)b.data(), &ldb, (double*)&beta, (double*)c.data(), &m);
    }
    else{
      blas::sgemm_(&transa, &transb, &m, &n, &k, (float*)&alpha, (float*)a.data(), &m,
                   (float*)b.data(), &ldb, (float*)&beta, (float*)c.data(), &m);
    }
  }; 
   




  //Matrix matrix multiply with output allocation
  static DenseMatrix<TPrecision> Multiply(DenseMatrix<TPrecision> &a, DenseMatrix<TPrecision> &b,
      bool transposeA = false, bool transposeB = false, TPrecision alpha = 1 ){

    int m;
    int n;

    if(transposeA){
      m = a.N();
    }
    else{
      m = a.M();
    }
    if(transposeB){
      n = b.M();
    }
    else{
      n = b.N();
    }

    DenseMatrix<TPrecision> c(m, n); 
    Linalg<TPrecision>::Multiply(a, b, c, transposeA, transposeB, alpha);
    return c;
  }; 
   


  
  //--- Matrix Vector multiplication methods  



  //Matrix Vector multiply with output allocation
  static DenseVector<TPrecision> Multiply(DenseMatrix<TPrecision> &a,
      DenseVector<TPrecision> &v, bool transpose = false, TPrecision alpha = 1){

    int lc;

    if(transpose){
      lc = a.N();
    }
    else{
      lc = a.M();
    }

    DenseVector<TPrecision> c(lc);
    Multiply(a, v, c, transpose, alpha);
    return c; 

  };

  //Matrix Vector multiply without output allocation
  static void Multiply(DenseMatrix<TPrecision> &a,
      DenseVector<TPrecision> &v, DenseVector<TPrecision> &out, 
      bool transpose = false, TPrecision alpha = 1){
  
    TPrecision beta = 0;

    char transa;
    int ma = a.M();
    int na = a.N();
    int lda = a.M();
    if(transpose){
      transa = 'T';
    }
    else{
      transa = 'N';
    }

    int inc = 1;
    if(isDoubleTPrecision()){
      blas::dgemv_(&transa,  &ma, &na, (double*)&alpha, (double*)a.data(),
          &lda, (double*)v.data(), &inc, (double*)&beta, (double*)out.data(), &inc);
    }
    else{
      blas::sgemv_(&transa,  &ma, &na, (float*)&alpha, (float*)a.data(),
          &lda, (float*)v.data(), &inc, (float*)&beta, (float*)out.data(), &inc);
    }
  }; 
 
 

  //Matrix vector (matrix column) multiply with output allocation 
  static DenseVector<TPrecision> MultiplyColumn(DenseMatrix<TPrecision> &a,
      DenseMatrix<TPrecision> &b, int index, bool transpose = false, TPrecision alpha = 1){

    int lc;

    if(transpose){
      lc = a.M();
    }
    else{
      lc = a.N();
    }

    DenseVector<TPrecision> c(lc);
    MultiplyColumn(a, b, index, c, transpose, alpha);
    return c; 
  }; 
  
  //Matrix vector (matrix column) multiply without output allocation 
  static void MultiplyColumn(DenseMatrix<TPrecision> &a,
      DenseMatrix<TPrecision> &b, int index, DenseVector<TPrecision> &out, 
      bool transpose = false, TPrecision alpha = 1){

    TPrecision beta = 0;

    char transa;
    if(transpose){
      transa = 'T';
    }
    else{
      transa = 'N';
    }
    int ma = a.M();
    int na = a.N();
    int lda = a.M();

    TPrecision *v = b.data();
    int vinc = 1;
    v = &v[index*b.M()];

    int inc = 1;

    if(isDoubleTPrecision()){
      blas::dgemv_(&transa,  &ma, &na, (double*)&alpha, (double*)a.data(),
          &lda, (double*)v, &vinc, (double*)&beta, (double*)out.data(), &inc);
    }
    else{
      blas::sgemv_(&transa,  &ma, &na, (float*)&alpha, (float*)a.data(),
          &lda, (float*)v, &vinc, (float*)&beta, (float*)out.data(), &inc);
    }
  }; 



  //Matrix vector (matrix row) multiply with output allocation 
  static DenseVector<TPrecision> MultiplyRow(DenseMatrix<TPrecision> &a,
      DenseMatrix<TPrecision> &b, int index, bool transpose = false, TPrecision alpha = 1){

    int lc;

      if(transpose){
        lc = a.M();
      }
      else{
        lc = a.N();
      }

    DenseVector<TPrecision> c(lc);
    MultiplyRow(a, b, index, c, transpose, alpha);
    return c; 
  }; 
  
  //Matrix vector (matrix row) multiply without output allocation 
  static void MultiplyRow(DenseMatrix<TPrecision> &a,
      DenseMatrix<TPrecision> &b, int index, DenseVector<TPrecision> &out, 
      bool transpose = false, TPrecision alpha = 1){

    TPrecision beta = 0;

    char transa;
    int ma = a.M();
    int na = a.N();
    if(transpose){
      transa = 'T';
    }
    else{
      transa = 'N';
    }
    int lda = a.M();

    
    TPrecision *v = b.data();
    int vinc = 1;
    if(b.isRowMajor()){
      v = &v[index*b.N];
    }
    else{
      v = &v[index];
      vinc = b.M();
    }


    int inc = 1;

    if(isDoubleTPrecision()){
      blas::dgemv_(&transa,  &ma, &na, (double*)&alpha, (double*)a.data(),
          &lda, (double*)v, &vinc, (double*)&beta, (double*)out.data(), &inc);
    }
    else{
      blas::sgemv_(&transa,  &ma, &na, (float*)&alpha, (float*)a.data(),
          &lda, (float*)v, &vinc, (float*)&beta, (float*)out.data(), &inc);
    }


  };






  //--- Dot prioduct methods


   static TPrecision Dot(DenseVector<TPrecision> &x, DenseVector<TPrecision> &y){
     
     int n = x.N();
     int inc = 1;
    
     TPrecision res = 0; 
     if( isDoubleTPrecision() ){
       res = (TPrecision) blas::ddot_(&n, (double *)x.data(), &inc, (double*)y.data(), &inc);
     }
     else{
       res = (TPrecision) blas::sdot_( &n, (float*)x.data(), &inc, (float*)y.data(), &inc);
     }
     return res;
   }; 

   

   static TPrecision DotColumn(DenseMatrix<TPrecision> &x, int index,
      DenseVector<TPrecision> &y){
    
      TPrecision *v = x.data();
      int vinc = 1;
      v = &v[index*x.M()];

     int n = y.N();
     int inc = 1;
    
     TPrecision res = 0; 
     if( isDoubleTPrecision() ){
       res = (TPrecision) blas::ddot_(&n, (double *)v, &vinc, (double*)y.data(), &inc);
     }
     else{
       res = (TPrecision) blas::sdot_(&n, (float*)v, &vinc, (float*)y.data(), &inc);
     }
     return res;
   }; 


   static TPrecision DotRow(DenseMatrix<TPrecision> &x, int index,
      DenseVector<TPrecision> &y){
    
      TPrecision *v = x.data();
      int vinc = 1;
      if(x.isRowMajor()){
        v = &v[index * x.N()];
      }
      else{
        v = &v[index];
        vinc = x.M();
      }

     int n = y.N();
     int inc = 1;
    
     TPrecision res = 0; 
     if( isDoubleTPrecision() ){
       res = (TPrecision) blas::ddot_(&n, (double*)v, &vinc, (double*)y.data(), &inc);
     }
     else{
       res = (TPrecision) blas::sdot_(&n, (float*)v, &vinc, (float*)y.data(), &inc);
     }
     return res;
   }; 


   static TPrecision DotRowRow(DenseMatrix<TPrecision> &x, int xindex,
      DenseMatrix<TPrecision> &y, int yindex ){
    
      TPrecision *v = x.data();
      int vinc = 1;
      if(x.isRowMajor()){
        v = &v[xindex * x.N()];
      }
      else{
        v = &v[xindex];
        vinc = x.M();
      }


      TPrecision *w = y.data();
      int winc = 1;
      if(y.isRowMajor()){
        w = &w[yindex * y.N()];
      }
      else{
        w = &w[yindex];
        winc = y.M();
      }


     int n = x.M();
    
     TPrecision res = 0; 
     if( isDoubleTPrecision() ){
       res = (TPrecision) blas::ddot_(&n, (double*)v, &vinc, (double*)w, &winc);
     }
     else{
       res = (TPrecision) blas::sdot_(&n, (float*)v, &vinc,(float*)w, &winc);
     }
     return res;
   }; 

   
   static TPrecision DotColumnRow(DenseMatrix<TPrecision> &x, int xindex,
      DenseMatrix<TPrecision> &y, int yindex ){
    return DotRowColumn(y, yindex, x, xindex);
   };


   static TPrecision DotRowColumn(DenseMatrix<TPrecision> &x, int xindex,
      DenseMatrix<TPrecision> &y, int yindex ){
    
      TPrecision *v = x.data();
      int vinc = 1;
      if(x.isRowMajor()){
        v = &v[xindex * x.N()];
      }
      else{
        v = &v[xindex];
        vinc = x.M();
      }


      TPrecision *w = y.data();
      int winc = 1;
      if(y.isRowMajor()){
        w = &v[yindex];
        winc = y.N();
      }
      else{
        w = &w[yindex*y.M()];
      }

     int n = x.M();
    
     TPrecision res = 0; 
     if( isDoubleTPrecision() ){
       res = (TPrecision) blas::ddot_(&n, (double*)v, &vinc, (double*)w, &winc);
     }
     else{
       res = (TPrecision) blas::sdot_(&n, (float*)v, &vinc,(float*)w, &winc);
     }
     return res;
   }; 

   
   static TPrecision DotColumnColumn(DenseMatrix<TPrecision> &x, int xindex,
      DenseMatrix<TPrecision> &y, int yindex ){
    
      TPrecision *v = x.data();
      int vinc = 1;

      v = &v[xindex*x.M()];

      TPrecision *w = y.data();
      int winc = 1;
      if(y.isRowMajor()){
        w = &v[yindex];
        winc = y.N();
      }
      else{
        w = &w[yindex*y.M()];
      }

     int n = x.M();
    
     TPrecision res = 0; 
     if( isDoubleTPrecision() ){
       res = (TPrecision) blas::ddot_(&n, (double*)v, &vinc, (double*)w, &winc);
     }
     else{
       res = (TPrecision) blas::sdot_(&n, (float*)v, &vinc,(float*)w, &winc);
     }
     return res;
   }; 









  //--------------- Methods for general matrices/vectors - slow ---------------// 




  //TODO
  /*static DenseMatrix<TPrecision> Multiply(Matrix<TPrecision> &a, Matrix<TPrecision> &b,
      bool transposeA = false, bool transposeB = false, TPrecision alpha = 0){

    DenseMatrix<TPrecision> c(a.M(), b.N(), false);
    
    return c; 
  };*/



  static DenseVector<TPrecision> ExtractColumn(Matrix<TPrecision> &a, int index){
    DenseVector<TPrecision> v(a.M());
    for(unsigned int i=0; i<a.M(); i++){
      v(i) = a(i, index);
    }
    return v;
  };
  



  static void ExtractColumn(Matrix<TPrecision> &a, int index, Vector<TPrecision> &v){
    for(unsigned int i=0; i<a.M(); i++){
      v(i) = a(i, index);
    }
  };
  



  static void ExtractRow(Matrix<TPrecision> &a, int index, Vector<TPrecision> &v){
    for(unsigned int i=0; i<a.N(); i++){
      v(i) = a(index, i);
    }
  };
  



  static void SetColumn(Matrix<TPrecision> &a, int index, Vector<TPrecision> &v){
    for(unsigned int i=0; i<a.M(); i++){
      a(i, index) = v(i);
    }
  };

  static void SetRow(Matrix<TPrecision> &a, int index, Vector<TPrecision> &v){
    for(unsigned int i=0; i<a.N(); i++){
      a(index, i) = v(i);
    }
  }; 


  static void SetColumn(Matrix<TPrecision> &a, int aindex,
      DenseMatrix<TPrecision> &b, int bindex){
    for(unsigned int i=0; i<a.M(); i++){
      a(i, aindex) = b(i, bindex);
    }
  };



  static void SetRow(Matrix<TPrecision> &a, int aindex,
      DenseMatrix<TPrecision> &b, int bindex){
    for(unsigned int i=0; i<a.N(); i++){
      a(aindex, i) = b(bindex, i);
    }
  };


  static void SetRowFromColumn(Matrix<TPrecision> &a, int aindex,
      DenseMatrix<TPrecision> &b, int bindex){
    for(unsigned int i=0; i<a.N(); i++){
      a(aindex, i) = b(i, bindex);
    }
  };


  static DenseVector<TPrecision> SumColumns(Matrix<TPrecision> &a){
    DenseVector<TPrecision> sum(a.M());
    SumColumns(a, sum);
    return sum;
  };

  


  static void SumColumns(Matrix<TPrecision> &a, Vector<TPrecision> &v){
      for(unsigned int i=0; i < a.M(); i++){
        v(i) = 0;
        for(unsigned int j=0; j < a.N(); j++){
          v(i) += a(i, j);
        }
      }
  };
  


  static DenseVector<TPrecision> SumRows(Matrix<TPrecision> &a){
    DenseVector<TPrecision> sum(a.N());
    SumRows(a, sum);
    return sum;
  };



  static void SumRows(Matrix<TPrecision> &a, Vector<TPrecision> &v){
      for(unsigned int j=0; j < a.N(); j++){
        v(j) = 0;
        for(unsigned int i=0; i < a.M(); i++){
          v(j) += a(i, j);
        }
      }
  };




  static void SubtractColumnwise(Matrix<TPrecision> &a, Vector<TPrecision> &v,
      Matrix<TPrecision> &out){
      for(unsigned int i=0; i < a.M(); i++){
        for(unsigned int j=0; j < a.N(); j++){
          out(i, j) = a(i, j) - v(i);
        }
      }
  };
  


  static void AddColumnwise(Matrix<TPrecision> &a, Vector<TPrecision> &v,
      Matrix<TPrecision> &out){
      for(unsigned int i=0; i < a.M(); i++){
        for(unsigned int j=0; j < a.N(); j++){
          out(i, j) = a(i, j) + v(i);
        }
      }
  };
  


  static void SubtractColumn(Matrix<TPrecision> &a, int index, Vector<TPrecision> &v,
      Matrix<TPrecision> &out){
      for(unsigned int i=0; i < a.M(); i++){
        out(i, index) = a(i, index) - v(i);
      }
  };



  static void SubtractRowwise(Matrix<TPrecision> &a, Vector<TPrecision> &v,
      Matrix<TPrecision> &out){
      for(unsigned int i=0; i < a.M(); i++){
        for(unsigned int j=0; j < a.N(); j++){
          out(i, j) = a(i, j) - v(j);
        }
      }
  };
  


  static void SubtractRow(Matrix<TPrecision> &a, int index, Vector<TPrecision> &v,
      Matrix<TPrecision> &out){
      for(unsigned int j=0; j < a.N(); j++){
        out(index, j) = a(index, j) - v(j);
      }
  };



  //result = a + b
  static void Add(Vector<TPrecision> &a, Vector<TPrecision> &b, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.N(); i++){
      result(i) = a(i) + b(i);
    }    
  };

  //result = a + B(:, index)
  static void Add(Vector<TPrecision> &a, Matrix<TPrecision> &B, int index, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.N(); i++){
      result(i) = a(i) + B(i, index);
    }    
  };


  //B(:, index) = a + B(:, index)
  static void Add(Matrix<TPrecision> &B, int index, Vector<TPrecision> &a){
    for(unsigned int i = 0; i < a.N(); i++){
      B(i, index) = a(i) + B(i, index);
    }    
  };
  
  //C = A + B
  static void Add(Matrix<TPrecision> &A, Matrix<TPrecision> &B, Matrix<TPrecision> &C){
    for(unsigned int i = 0; i < A.N(); i++){
      for(unsigned int j = 0; j< A.M(); j++){
        C(j, i) = A(j, i) + B(j, i);
      }
    }    
  };    
  
  
  //C = A - B
  static void Subtract(Matrix<TPrecision> &A, Matrix<TPrecision> &B, Matrix<TPrecision> &C){
    for(unsigned int i = 0; i < A.N(); i++){
      for(unsigned int j = 0; j< A.M(); j++){
        C(j, i) = A(j, i) - B(j, i);
      }
    }    
  };     
  
  static void Scale(Matrix<TPrecision> &A, TPrecision s, Matrix<TPrecision> &C){
    for(unsigned int i = 0; i < A.N(); i++){
      for(unsigned int j = 0; j< A.M(); j++){
        C(j, i) = s*A(j, i);
      }
    }    
  };
  
  //result = a + s*b
  static void AddScale(Vector<TPrecision> &a, TPrecision s, Vector<TPrecision> &b, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.N(); i++){
      result(i) = a(i) + s * b(i);
    }    
  };
 


  static void AddScale(Vector<TPrecision> &a, TPrecision s, Matrix<TPrecision> &b,
      int index, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.N(); i++){
      result(i) = a(i) + s * b(i, index);
    }    
  };

  
  static void AddScale(Matrix<TPrecision> &a, TPrecision s, Matrix<TPrecision> &b, Matrix<TPrecision> &result){
    for(unsigned int i = 0; i < a.M(); i++){
      for(unsigned int j = 0; j < a.N(); j++){
        result(i, j) = a(i, j) + s * b(i, j);
      }
    }    
  };

  static void SubtractScale(Matrix<TPrecision> &a, TPrecision s, Matrix<TPrecision> &b, Matrix<TPrecision> &result){
    for(unsigned int i = 0; i < a.M(); i++){
      for(unsigned int j = 0; j < a.N(); j++){
        result(i, j) = a(i, j) - s * b(i, j);
      }
    }    
  };



  
  static void ColumnAddScale(DenseMatrix<TPrecision> &m, int index, TPrecision s, Vector<TPrecision> &b){
    for(unsigned int i = 0; i < m.M(); i++){
      m(i, index) = m(i, index) + s * b(i);
    }    
  };



  static DenseVector<TPrecision> ColumnwiseSquaredNorm(Matrix<TPrecision> &A){
    DenseVector<TPrecision> norms(A.N());
    ColumnwiseSquaredNorm(A, norms);
    return norms;
  };
	  
  static void ColumnwiseSquaredNorm(Matrix<TPrecision> &A, Vector<TPrecision> &v){	  
    for(unsigned int i=0; i< A.N(); i++){
      TPrecision norm =0;
      TPrecision tmp =0;
      for(unsigned int j=0; j<A.M(); j++){
	tmp = A(j, i);
        norm += tmp *tmp;
      }
      v(i) = norm;
    }       
  };


  static DenseVector<TPrecision> ColumnwiseNorm(Matrix<TPrecision> &A){
    DenseVector<TPrecision> norms(A.N());
    ColumnwiseNorm(A, norms);
    return norms;
  };
	  
  static void ColumnwiseNorm(Matrix<TPrecision> &A, Vector<TPrecision> &v){
    for(unsigned int i=0; i< A.N(); i++){
      TPrecision norm =0;
      TPrecision tmp =0;
      for(unsigned int j=0; j<A.M(); j++){
	tmp = A(j, i);
        norm += tmp *tmp;
      }
      v(i) = sqrt(norm);
    }       
  };



  //result = a - b
  static void Subtract(Vector<TPrecision> &a, TPrecision b, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.N(); i++){
      result(i) = a(i) - b;
    }    
  };
  //result = a - b
  static void Add(Vector<TPrecision> &a, TPrecision b, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.N(); i++){
      result(i) = a(i) + b;
    }    
  };




  //result = a - b
  static void Subtract(Vector<TPrecision> &a, Vector<TPrecision> &b, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.N(); i++){
      result(i) = a(i) - b(i);
    }    
  };



  //returns a - b
  static DenseVector<TPrecision> Subtract(Vector<TPrecision> &a, Vector<TPrecision> &b){
    DenseVector<TPrecision> result(a.N());
    Subtract(a, b, result);    
    return result;   
  };



  //result = a(:, index) - b
  static void Subtract(Matrix<TPrecision> &a, int index, Vector<TPrecision> &b, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.M(); i++){
      result(i) = a(i, index) - b(i);
    }    
  };
  


  //result =b - a(:, index)
  static void Subtract(Vector<TPrecision> &b, Matrix<TPrecision> &a, int index, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.M(); i++){
      result(i) = b(i) - a(i, index);
    }    
  };



  //result = a(:, index) - b(:, index)
  static void Subtract(Matrix<TPrecision> &a, int aindex, Matrix<TPrecision> &b,
      int bindex, Vector<TPrecision> &result){
    for(unsigned int i = 0; i < a.M(); i++){
      result(i) = a(i, aindex) - b(i, bindex);
    }    
  };

  static DenseMatrix<TPrecision> Copy(DenseMatrix<TPrecision> &from){
    DenseMatrix<TPrecision> to(from.M(), from.N());
    Copy(from, to);
    return to;
  };



  static void Copy(Matrix<TPrecision> &from, Matrix<TPrecision> &to){
    for(unsigned int i = 0; i < from.N(); i++){
      for(unsigned int j= 0; j < from.M(); j++){
        to(j, i) = from(j, i);
      }
    } 
  };  
 

  static void CopyColumn(Matrix<TPrecision> &from, unsigned int fi, Matrix<TPrecision>
      &to, unsigned int ti){
    for(unsigned int j= 0; j < from.M(); j++){
      to(j, ti) = from(j, fi);
    }
  };



  static DenseVector<TPrecision> Copy(Vector<TPrecision> &from){
    DenseVector<TPrecision> to(from.N());
    Copy(from, to);
    return to;
  };




  static void Copy(Vector<TPrecision> &from, Vector<TPrecision> &to){
    for(unsigned int i = 0; i < from.N(); i++){
      to(i) = from(i);
    } 
  };


  static TPrecision SquaredLengthColumn(Matrix<TPrecision> &v, int index){
    TPrecision l = 0;
    for(unsigned int i = 0; i < v.M(); i++){
      l += v(i, index) * v(i, index);
    }  
    return l;
  };

  static TPrecision LengthColumn(Matrix<TPrecision> &v, int index){
    TPrecision l = 0;
    for(unsigned int i = 0; i < v.M(); i++){
      l += v(i, index) * v(i, index);
    }  
    return sqrt(l);
  };

  static TPrecision LengthRow(Matrix<TPrecision> &v, int index){
    TPrecision l = 0;
    for(unsigned int i = 0; i < v.N(); i++){
      l += v(index, i) * v(index, i);
    }  
    return sqrt(l);
  };

  static TPrecision Length(Vector<TPrecision> &v){
    TPrecision l = 0;
    for(unsigned int i = 0; i < v.N(); i++){
      l += v(i) * v(i);
    }  
    return sqrt(l);
  };
  
  static TPrecision SquaredLength(Vector<TPrecision> &v){
    TPrecision l = 0;
    for(unsigned int i = 0; i < v.N(); i++){
      l += v(i) * v(i);
    }  
    return l;
  };



  static TPrecision Dot(Vector<TPrecision> &a, Vector<TPrecision> &b){
    TPrecision tmp = 0;
    for(unsigned int i = 0; i < a.N(); i++){
      tmp += b(i) * a(i);
    }  
    return tmp; 
  };

  static DenseMatrix<TPrecision> OuterProduct(Vector<TPrecision> &a, Vector<TPrecision> &b){
    DenseMatrix<TPrecision> op(a.N(), b.N());
    OuterProduct(a, b, op);
    return op;
  };


  static void OuterProduct(Vector<TPrecision> &a,
    Vector<TPrecision> &b, Matrix<TPrecision> &op){
    for(unsigned int i=0; i<a.N(); i++){
      for(unsigned int j=0; j<b.N(); j++){
        op(j,i) = a(i) * b(j);
      }
    }
  };  
  
  static void AddOuterProduct(Matrix<TPrecision> &c, Vector<TPrecision> &a,
    Vector<TPrecision> &b, Matrix<TPrecision> &op){
    for(unsigned int i=0; i<a.N(); i++){
      for(unsigned int j=0; j<b.N(); j++){
        op(j,i) = c(j, i) + a(i) * b(j);
      }
    }
  };  
  
  static void SubtractOuterProduct(Matrix<TPrecision> &c, Vector<TPrecision> &a,
    Vector<TPrecision> &b, Matrix<TPrecision> &op){
    for(unsigned int i=0; i<a.N(); i++){
      for(unsigned int j=0; j<b.N(); j++){
        op(j,i) = c(j, i) - a(i) * b(j);
      }
    }
  };  
  
  static void AddOuterProduct(Matrix<TPrecision> &c, Matrix<TPrecision> &a,
      unsigned int ai, Matrix<TPrecision> &b, unsigned int bi, 
      Matrix<TPrecision> &op){

    for(unsigned int i=0; i<a.M(); i++){
      for(unsigned int j=0; j<b.M(); j++){
        op(j,i) = c(j,i) + a(i, ai) * b(j, bi);
      }
    }
  };  
  
  static void AddOuterProduct(Matrix<TPrecision> &c, Vector<TPrecision> &a, 
      Matrix<TPrecision> &b, unsigned int bi, Matrix<TPrecision> &op){

    for(unsigned int i=0; i<a.N(); i++){
      for(unsigned int j=0; j<b.M(); j++){
        op(j,i) = c(j,i) + a(i) * b(j, bi);
      }
    }
  };






  //result = v*s
  static void Scale(Vector<TPrecision> &v, TPrecision s, Vector<TPrecision> &result){
    for(unsigned int i=0; i < v.N(); i++){
      result(i) = v(i)*s;
    }
  };   
  

  static void Sqrt(Vector<TPrecision> &v, Vector<TPrecision> &result){
    for(unsigned int i=0; i < v.N(); i++){
      result(i) = sqrt(v(i));
    }
  };   
  
  //returns 0 vector for vectors with 1/length = inf
  static void Normalize(Vector<TPrecision> &v){
      TPrecision l = 1.0/Length(v);
      if( l!=l || 
          std::numeric_limits<TPrecision>::infinity() == l ){
        l = 0;
      }
      Scale(v, l, v);
  }; 
  



  static void ScaleColumn(DenseMatrix<TPrecision> &m, int index, TPrecision s){
    for(unsigned int i=0; i < m.M(); i++){
      m(i,index) = m(i, index) * s;
    }
  };

  static void ScaleRow(DenseMatrix<TPrecision> &m, int index, TPrecision s){
    for(unsigned int i=0; i < m.N(); i++){
      m(index, i) = m(index, i) * s;
    }
  };



  static void Set(Vector<TPrecision> &v, TPrecision s){
    for(unsigned int i=0; i < v.N(); i++){
      v(i) = s;
    }
  };
  


  static void Set(Matrix<TPrecision> &a, TPrecision s){
    for(unsigned int i=0; i<a.M(); i++){
      for(unsigned int j=0; j<a.N(); j++){
        a(i, j) = s;
      }
    }
  };



  static void Zero(DenseMatrix<TPrecision> &m){
    memset(m.data(), 0, m.N() * m.M() * sizeof(TPrecision));
  };
  


  static void Zero(DenseVector<TPrecision> &v){
    memset(v.data(), 0, v.N() * sizeof(TPrecision));
  };
  
  static void Zero(Vector<TPrecision> &v){
    Set(v, 0);
  };  
  
  static void Zero(Matrix<TPrecision> &m){
    Set(m, 0);    
  };
  





  static void Scale(DenseMatrix<TPrecision> &m, TPrecision s,
      DenseMatrix<TPrecision> &r){
    TPrecision *a = m.data();
    TPrecision *b = r.data();
    unsigned int l = m.N() * m.M();
    for(unsigned int i=0; i<l; i++){
      b[i] = a[i]*s;
    }
  };



  static TPrecision FrobeniusNorm(Matrix<TPrecision> &m){
    return sqrt(SquaredFrobeniusNorm(m));
  };

  static TPrecision SquaredFrobeniusNorm(Matrix<TPrecision> &m){
    TPrecision tmp = 0;
    TPrecision norm = 0;
    for(unsigned int i=0; i<m.M(); i++){
      for(unsigned int j=0; j<m.N(); j++){
        tmp = m(i, j);
        norm += tmp*tmp;
      }
    }
    return norm;
  };

  
  static DenseMatrix<TPrecision> Identity(int n){
    DenseMatrix<TPrecision> eye(n, n);
    eye.zero();
    for(unsigned int i=0; i<n; i++){
      eye(i, i) = 1;
    }
    return eye;
  };

  static DenseMatrix<TPrecision> Transpose(DenseMatrix<TPrecision> A){
    DenseMatrix<TPrecision> B(A.N(), A.M());
    for(unsigned int i=0; i<A.M(); i++){
      for(unsigned int j=0; j<A.N(); j++){
        B(j, i) = A(i, j);
      }
    }
    return B;
  };



  static bool IsColumnEqual(DenseMatrix<TPrecision> A, int ai, DenseMatrix<TPrecision>
      B, int bi){
    bool equal = true;
    for(unsigned int i=0; i< A.M() && equal; i++){
      equal = A(i, ai) == B(i, bi);
    }
    return equal;
  };



  static DenseVector<TPrecision> Max(Matrix<TPrecision> &A){
    DenseVector<TPrecision> m = Linalg<TPrecision>::ExtractColumn(A, 0);
    for(unsigned  int i=0;i<A.N(); i++){
      for(unsigned int j=0; j<A.M(); j++){
        if(A(j, i) > m(j)){
          m(j) = A(j, i);
        }
      }
    }
    return m;
  };




  static DenseVector<TPrecision> Min(Matrix<TPrecision> &A){
    DenseVector<TPrecision> m = Linalg<TPrecision>::ExtractColumn(A, 0);
    for(unsigned int i=0;i<A.N(); i++){
      for(unsigned int j=0; j<A.M(); j++){
        if(A(j, i) < m(j)){
          m(j) = A(j, i);
        }
      }
    }
    return m;
  };


  static TPrecision Max(Vector<TPrecision> &A){
    TPrecision m = A(0);
    for(unsigned int i=0;i<A.N(); i++){
        if(A(i) > m){
          m = A(i);
        }
    }
    return m;
  };
  
  static TPrecision Min(Vector<TPrecision> &A){
    TPrecision m = A(0);
    for(unsigned int i=0;i<A.N(); i++){
        if(A(i) < m){
          m = A(i);
        }
    }
    return m;
  };


};

#endif
