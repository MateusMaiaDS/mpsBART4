// #include <random>
// #include <RcppArmadillo.h>
// using namespace std;
//
// //[[Rcpp::export]]
// arma::mat sum_exclude_col(arma::mat mat, int exclude_int){
//
//      // Setting the sum matrix
//      arma::mat m(mat.n_rows,1);
//
//      if(exclude_int==0){
//           m = sum(mat.cols(1,mat.n_cols-1),1);
//      } else if(exclude_int == (mat.n_cols-1)){
//           m = sum(mat.cols(0,mat.n_cols-2),1);
//      } else {
//           m = arma::sum(mat.cols(0,exclude_int-1),1) + arma::sum(mat.cols(exclude_int+1,mat.n_cols-1),1);
//      }
//
//      return m;
// }
//
// //[[Rcpp::export]]
// double inn_prod_one(arma::mat beta_vec){
//      arma::vec sum = beta_vec.t()*beta_vec;
//      return sum(0,0) ;
// }
//
// //[[Rcpp::export]]
// double inn_prod_two(arma::mat beta_vec){
//      return arma::accu(beta_vec*beta_vec);
// }
//
// //[[Rcpp::export]]
// double inn_prod_three(arma::mat beta_vec){
//      return arma::accu(arma::square(beta_vec));
// }
//
// //[[Rcpp::export]]
// void array_check(arma::cube arr_){
//      cout << "Slices: " << arr_.n_slices << endl;
//      cout << "Columns: " << arr_.n_cols << endl;
//      cout << "Rows: " << arr_.n_rows << endl;
//
//      return;
// }
//
//
