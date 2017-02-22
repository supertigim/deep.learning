#include <assert.h>
#include "Matrix.h"
#include <ctime>

template<typename T>
void Matrix<T>::initialize (const int& m, const int& n, const bool init)  {

	const int num_all_old = num_rows_ * num_cols_;

	num_rows_ = m;
	num_cols_ = n;

	SAFE_DELETE_ARRAY(values_);

	const int num_all = num_rows_ * num_cols_;

	if (num_all_old != num_all) {

		assert((double)num_rows_ * (double)num_cols_ <= (double)INT_MAX);

		values_ = new T[num_all];

		if(init == true) {
			for (int i = 0 ; i < num_all ; ++i){
				values_[i] = (T)0;
			}
		}
	}
}

template<typename T>
void Matrix<T>::assignRandom(const T& scale, const T& min)
{
	const int num_all = num_rows_ * num_cols_;
	
	//srand((unsigned int)time(0));
	for (int i = 0; i < num_all; ++i){
		values_[i] = (T)rand() / (T)RAND_MAX * scale + min;
	}
}

template<typename T>
void Matrix<T>::assignAll(const T& v)
{
	const int num_all = num_rows_ * num_cols_;
	for (int i = 0; i < num_all; ++i)
		values_[i] = v;
}

template<typename T>
void Matrix<T>::multiply(const VectorND<T>& vec,  VectorND<T>& result) const {

	assert(num_rows_ <= result.num_dimension_);
	assert(num_cols_ <= vec.num_dimension_);

	for( int row = 0 ; row < num_rows_ ; ++row) {
		result.values_[row] = (T)0;

		int ix = row*num_cols_;
		T temp;

		for( int col = 0 ; col < num_cols_ ; ++col){
			temp = values_[ix + col];
			temp *= vec.values_[col];

			result.values_[row] += temp;
		}
	}
}

template<typename T>
void Matrix<T>::multiplyTransposed(const VectorND<T>& vec,  VectorND<T>& result) const {
	assert(num_rows_ <= vec.num_dimension_);
	assert(num_cols_ <= result.num_dimension_);

	for(int col = 0 ; col < num_cols_ ; ++col) {
		result.values_[col] = (T)0;

		for (int row = 0, ix = col ; row < num_rows_; ++row, ix += num_cols_)
		{
			result.values_[col] += values_[ix] * vec.values_[row];
		}
	}
}

template<typename T>
int Matrix<T>::get1DIndex(const int& row, const int& col) const {
	assert( row >= 0 );
	assert( col >= 0 );
	assert( row < num_rows_ );
	assert( col < num_cols_ ); //?

	return col + row * num_cols_;
}

template<typename T>
T& Matrix<T>::getValue(const int& row, const int& col) const {
	return values_[get1DIndex(row,col)];
}

template<typename T>
void Matrix<T>::setDiagonal() {
	const int num = MIN2(num_cols_, num_rows_);

	for (int i = 0; i < num_cols_ * num_rows_; ++i) {
		values_[i] = 0.0;
	}

	for (int i = 0; i < num; ++i) {
		getValue(i, i) = 1.0;
	}
}

template<typename T>
void Matrix<T>::cout() {
    for (int row = 0; row < num_rows_; ++row) {
        for (int col = 0; col < num_cols_; ++col) {
            std::cout << getValue(row, col) << " ";
        }
        std::cout << endl;
    }
}

template<typename T>
void Matrix<T>::normalizeAllRows(const T& row_sum_min) {
	for (int row = 0; row < num_rows_; ++row) {
		normalizeRow(row, row_sum_min);
	}
}

template<typename T>
void Matrix<T>::normalizeRow(const int& row, const T& row_sum_min) {
	T row_sum = (T)0;

	for (int col = 0; col < num_cols_-1; ++col) {
		// TODO normalize bias option
		row_sum += getValue(row, col);
	}

	if (row_sum > row_sum_min) {
		for (int col = 0; col < num_cols_-1; ++col) {
			// TODO normalize bias option
			getValue(row, col) /= row_sum;
		}
	}
}

template<typename T>
void Matrix<T>::writeTXT(std::ofstream& of) const {
	of << num_rows_ << " " << num_cols_ << endl;
	for (int i = 0; i < num_rows_ * num_cols_; ++i) {
		if (i != 0 && i % num_cols_ == 0) of << endl;
		of << values_[i];

		if (i != num_rows_ * num_cols_ - 1) {
			of << " ";			
		}
	}
	of << endl;
}

template<typename T>
void Matrix<T>::check() const {
	for (int i = 0; i < num_rows_ * num_cols_; ++i) {
		if (std::isnan(values_[i])) {
			std::cout << "Nan " << values_[i] << endl;
			assert(false);
			exit(1);
		}

		if (std::isinf(values_[i])) {
			std::cout << "inf " << values_[i] << endl;
			assert(false);
			exit(1);
		}
	}
}

// to prevent link error. dummies...
template class Matrix<float>;
template class Matrix<double>;


// end of file 
