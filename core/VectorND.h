#pragma once 

#include "Common.h"
#include <assert.h>
#include <iostream>		
#include <fstream>		// for ofstream
#include <cmath>		// for std::isnan and std::isnaf


template <typename T>
class VectorND {
public:
	int num_dimension_;
	T *values_;

public:
	VectorND()
		: num_dimension_(0), values_(nullptr)
	{}

	VectorND(const int& num){

		values_ = nullptr;
		initialize(num);
	}

	VectorND(const VectorND<T>& vec){
		values_ = nullptr;

		initialize(vec.num_dimension_, false);

		for( int i = 0 ; i < num_dimension_; ++i) {
			values_[i] = vec[i];
		}
	}

	~VectorND() {
		if(values_ != nullptr) delete [] values_;
		num_dimension_ = 0;
	}

public:

	void initialize(const int& num, const bool init = false) {

		num_dimension_ = num;
		SAFE_DELETE(values_);

		if(num_dimension_ > 0) {
			values_ = new T [num_dimension_];
			if(init == true) {
				for(int i = 0 ; i < num_dimension_ ; ++i) {
					values_[i] = T();
				}
			}

		}
	}

	void operator = (const VectorND<T>& from) {
		if (from.num_dimension_ != num_dimension_){
			num_dimension_ = from.num_dimension_;
			SAFE_DELETE(values_);
			values_ = new T [num_dimension_];
		}

		for (int i = 0 ; i < num_dimension_ ; ++i) {
			values_[i] = from[i];
		}
	}

	T& operator [] (const int& i) const {
		assert(i >= 0);
		assert(i < num_dimension_);

		return values_[i];
	}

	T& operator () (const int& i) const {
		assert(i >= 0);
		assert(i < num_dimension_);

		return values_[i];	
	}

	VectorND<T> operator + (const VectorND<T> & vec) const {
		assert(num_dimension_ == vec.num_dimension_);

		VectorND<T> result(num_dimension_);

		for (int i = 0; i < num_dimension_ ; ++i) {
			result[i] = values_[i] + vec.values_[i];
		}

		return result;
	}

	VectorND<T> operator - (const VectorND<T> & vec) const {
		assert(num_dimension_ == vec.num_dimension_);

		VectorND<T> result(num_dimension_);

		for (int i = 0; i < num_dimension_ ; ++i) {
			result[i] = values_[i] - vec.values_[i];
		}

		return result;
	}

	void operator += (const T& val){
		for (int i = 0 ; i < num_dimension_ ; ++i){
			values_[i] += val;
		}
	}

	void operator -= (const T& val){
		for (int i = 0 ; i < num_dimension_ ; ++i){
			values_[i] -= val;
		}
	}

	void operator *= (const int& val){
		for (int i = 0 ; i < num_dimension_ ; ++i){
			values_[i] *= val;
		}
	}

	void operator *= (const T& val){
		for (int i = 0 ; i < num_dimension_ ; ++i){
			values_[i] *= val;
		}
	}

	void operator /= (const T& val){
		for (int i = 0 ; i < num_dimension_ ; ++i){
			values_[i] /= val;
		}
	}

	void operator += (const VectorND<T>& vec){
		assert(num_dimension_ == vec.num_dimension_);

		for (int i = 0 ; i < num_dimension_ ; ++i){
			values_[i] += vec.values_[i];
		}	
	}

	void operator -= (const VectorND<T>& vec){
		assert(num_dimension_ == vec.num_dimension_);
		
		for (int i = 0 ; i < num_dimension_ ; ++i){
			values_[i] -= vec.values_[i];
		}	
	}

	void operator *= (const VectorND<T>& vec){
		assert(num_dimension_ == vec.num_dimension_);
		
		for (int i = 0 ; i < num_dimension_ ; ++i){
			values_[i] *= vec.values_[i];
		}	
	}

	void operator /= (const VectorND<T>& vec){
		assert(num_dimension_ == vec.num_dimension_);
		
		for (int i = 0 ; i < num_dimension_ ; ++i){
			values_[i] /= vec.values_[i];
		}	
	}

	VectorND<T> operator * (const T& val) const {

		VectorND<T> V(num_dimension_);

		for (int i = 0; i < num_dimension_ ; ++i) {
			V.values_[i] = values_[i] * val;
		}

		return V;
	}


	void assignAllValues(const T& input) {
		for (int i = 0; i < num_dimension_; ++i) {
			values_[i] = input;	
		} 
	}

	void copyPartial(const VectorND<T>& src, const int& start_ix_this, const int& start_ix_src, const int num) {
		assert(start_ix_this >= 0);
		assert(start_ix_this + num <= num_dimension_);

		assert(start_ix_src >= 0);
		assert(start_ix_src +num <= src.num_dimension_);

		for (int i = 0 ; i < num ; ++i) {
			values_[start_ix_this + i] =  src.values_[start_ix_src + i];
		}
	}

/*
	void write(std::ofstream& os) const {
		os.write((char*)&i_start_, sizeof(i_start_));
		os.write((char*)&j_start_, sizeof(j_start_));
		os.write((char*)&k_start_, sizeof(k_start_));

		os.write((char*)&i_res_, sizeof(i_res_));
		os.write((char*)&j_res_, sizeof(j_res_));
		os.write((char*)&k_res_, sizeof(k_res_));

		for (int p = 0; p < num_dimension_; ++p) {
			os.write((char*)&values_[p], sizeof(T));
		}
	}
*/

	void writeTXT(std::ofstream& os) const {
		os << num_dimension_ << endl;
		for (int i = 0; i < num_dimension_; ++i) {
			os << values_[i];

			if (i != num_dimension_ - 1)
				os << " ";
		}
		os << endl;
	}

	void check() const {
		for (int i = 0; i < num_dimension_; ++i) {
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

	T getMaxValue()
	{
		assert(num_dimension_ > 0);
		T max = values_[0];
		int ix = 0;

		for (int d = 1; d < num_dimension_; ++d) {
			if (max < values_[d]) {
				max = values_[d];
				ix = d;
			}
		}
		return max;
	}
};

template<class T>
void dotProduct(const VectorND<T>& v1, const VectorND<T>& v2, T& sum){

	assert (v1.num_dimension_ == v2.num_dimension_);
	sum = 0;

	for (int i = 0 ; i < v1.num_dimension_ ; ++i){
		sum += v1.values_[i] * v2.values_[i];
	}
}

template <class T>
std::ostream& operator << (std::ostream& output, const VectorND<T>& v) {
	for(int i = 0; i < v.num_dimension_ ; ++i) {
		output << v.values_[i] << " ";
	}

	output << std::flush ;
	return output;
}


// end of file
