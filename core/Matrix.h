#pragma once 

#include "VectorND.h"

template<typename T>
class Matrix{
public:
	T *values_;
	int num_rows_;
	int num_cols_;	

public:
	Matrix()
		: values_(nullptr), num_rows_(0), num_cols_(0)
	{}

	void initialize (const int& m, const int& n, const bool init = true) ;

	void assignRandom(const T& scale, const T& min);
	void assignAll(const T& v);

	void multiply(const VectorND<T>& vec,  VectorND<T>& result) const;
	void multiplyTransposed(const VectorND<T>& vec,  VectorND<T>& result) const;

	int get1DIndex(const int& row, const int& column) const;
	T& getValue(const int& row, const int& column) const;

	void cout();
	void setDiagonal();
	void normalizeAllRows(const T& row_sum_min);
	void normalizeRow(const int& row, const T& row_sum_min);
	void writeTXT(std::ofstream& of) const;
	void check() const;
};


// end of file
