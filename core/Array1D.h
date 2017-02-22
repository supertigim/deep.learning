#pragma once 

#include "Common.h"
#include <iostream>
#include <fstream>
#include <assert.h>

template<typename T>
class Array1D {

public:
	int num_elements_;
	T *values_;

	Array1D()
		: num_elements_(0), values_(nullptr)
	{}

	Array1D(const int& num_elements_input)
		: num_elements_(0), values_(nullptr)
	{
		initialize(num_elements_input);
	}

	Array1D(const int& num_elements_input, const T& values_input)
		: num_elements_(0), values_(nullptr)
	{
		initialize(num_elements_input, values_input);
	}

	Array1D(const Array1D<T>& array_input) 
		: num_elements_(0), values_(nullptr)
	{
		initialize(array_input);
	}

	~Array1D(){

		freeMemory();
		//SAFE_DELETE_ARRAY(values_);
		//num_elements_ = 0;
	}

	void initialize(const int& num_elements_input) {
		num_elements_ = num_elements_input;
		SAFE_DELETE_ARRAY(values_);
		if(num_elements_ > 0) {
			values_ = new T[num_elements_];
		}
	}

	void initialize(const int& num_elements_input, const T& values_input) {
		num_elements_ = num_elements_input;

		SAFE_DELETE_ARRAY(values_);

		if(num_elements_ > 0){
			values_ = new T[num_elements_];

			assignAllValues(values_input);
		}
	}

	void initialize(const Array1D<T>& array_input){
		initialize(array_input.num_elements_);

		copyFrom(array_input);
	}

	void assignAllValues(const T& constant){
		for (int i = 0 ;  i < num_elements_ ;  ++i) {
			values_[i] = constant;
		}
	}

	void assignValues (const int& start_ix, const int& end_ix, const T& constant) {
		for(int i = start_ix; i <= end_ix ; ++i) {
			values_[i] = constant;
		}
	}

	void assignMin(const int& ix, const T& value) {
		T& temp = *(values_ + ix);
		temp = MIN2(value, temp);
	}

	void copyFrom(const Array1D<T>& from) {
		assert(num_elements_ == from.num_elements_);

		T *from_val = from.values_;

		for(int i = 0 ; i < num_elements_ ; ++i) {
			values_[i] = from_val[i];
		}
	}

	void resize (const int new_size) {
		T *new_values = new T[new_size];

		for( int i = 0 ; i < MIN2(new_size, num_elements_); ++i) {
			new_values[i] = values_[i];
		}

		SWAP(values_, new_values, T);

		SAFE_DELETE_ARRAY(new_values);

		num_elements_ = new_size;
	}

	void freeMemory() {
		num_elements_ = 0 ;
		SAFE_DELETE_ARRAY(values_);
	}

	T& operator [] (const int& i) const {
		assert( i >= 0 && i < num_elements_ );

		return values_[i];
	}

	const int getSizeOfData() const {
		return num_elements_ * sizeof(T);
	}

	const int getSizeOfType() const {
		return sizeof(T);
	}

	void compactArray(Array1D<int>& flag_array, const int remove_flag) {
		int new_num_elements = 0;
		for(int i = 0 ; i < num_elements_ ; i++) {
			if( flag_array[i] != remove_flag) 
				flag_array[i] = new_num_elements++;
		}

		static Array1D<T> temp;
		temp.initialize(new_num_elements);	// for multi-threading

		for(int i = 0; i < num_elements_ ; ++i) {
			const int new_ix = flag_array[i];

			if(new_ix == remove_flag) continue;

			temp[new_ix] = values_[i];
		}

		//swap
		num_elements_ = temp.num_elements_;
		delete [] values_;
		values_ = temp.values_;

		// is it necessary?
		temp.num_elements_ = 0;
		temp.values_ = nullptr;
	}

	void copyToPartialArray(Array1D<T>& dest, const int& start, const int& end) {
		dest.initialize( end - start + 1 );

		for( int ix = 0, i = start ; i < end ; ++i, ++ix) {
			dest.values_[ix] = values_[i];
		}
	}

	void append(const Array1D<T>& src) {
		const int num_temp = num_elements_;

		resize(num_elements_ + src.num_elements_);

		for( int i = num_temp ; i < num_elements_ ; ++i) {
			values_[i] = src.values_[i - num_temp];
		}
	}

	int searchIncremental(const T& v) {
		for(int i = 0 ; i< num_elements_ ; ++i) {
			if( v < values_[i]) return i - 1;
		}

		return num_elements_;
	}

	void accumulateFromLeft(){
		T acc = (T)0;

		for(int i = 0 ; i < num_elements_; ++i) {
			values_[i] += acc;
			acc = values_[i];
		}
	}

	friend std::ostream& operator << (std::ostream& stream, const Array1D<T>& arr) {
		for(int i = 0 ; i < arr.num_elements_ ; ++i) {
			stream << arr[i] << " ";
		}

		return stream;
	}

	void read(std::ifstream& is){
		int num_elements;

		is.read((char*)&num_elements, sizeof(num_elements));


		initialize(num_elements);

		for(int i = 0 ; i < num_elements ; ++i) {
			is.read((char*)&values_[i], sizeof(T));
		}

	}

	void write(std::ofstream& os) const {
		os.write((char*)&num_elements_, sizeof(num_elements_));

		for(int i =0 ; i < num_elements_ ; ++i) {
			os.write((char*)&values_[i], sizeof(T));
		}
	}

	void operator *= (const T& constant) {
		for (int i = 0; i < num_elements_; i++) values_[i] *= constant;
	}

	void operator += (const T& constant) {
		for (int i = 0; i < num_elements_; i++) values_[i] += constant;
	}

	void operator -= (const T& constant) {
		for (int i = 0; i < num_elements_; i++) values_[i] -= constant;
	}

};

// end of file
