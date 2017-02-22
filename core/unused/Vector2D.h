#pragma once

#include <iostream>
#include <cmath>

template<typename T>
class Vector2D
{
public:
	union{
		struct{ T x_, y_; };
		struct{ T i_, j_; };
		struct{ T u_, v_; };
		struct{ T t_min_, t_max_; };
		struct{ T v0_, v1_; };
		T values_[2];
	};	

public:
	Vector2D(void)
		: x_(T()),y_(T())
	{}

	Vector2D(const T& x_input,const T& y_input)
		: x_(x_input), y_(y_input)
	{};

	Vector2D(const Vector2D& vector_input)
		: x_(vector_input.x_), y_(vector_input.y_)
	{}

	Vector2D(const T values_input[2])
		: x_(values_input[0]), y_(values_input[1])
	{}	

	~Vector2D(void)
	{};

public:
	void operator = (const Vector2D& v)
	{
		x_ = v.x_;
		y_ = v.y_;
	}

	void operator += (const Vector2D& v)
	{
		x_ += v.x_;
		y_ += v.y_;
	}

	void operator -= (const Vector2D& v)
	{
		x_ -= v.x_;
		y_ -= v.y_;
	}

	void operator *= (const T& s)
	{
		x_ *= s;
		y_ *= s;
	}

	void operator /= (const T& s)
	{
		const T one_over_s = (T)1/s;
		x_ *= one_over_s;
		y_ *= one_over_s;
	}

	Vector2D<T> operator + (const Vector2D<T>& v) const
	{
		return Vector2D<T>(x_+v.x_, y_+v.y_);
	}

	Vector2D<T> operator - (const Vector2D<T>& v) const
	{
		return Vector2D<T>(x_-v.x_, y_-v.y_);
	}

	Vector2D<T> operator * (const T& a) const
    {
		return Vector2D<T>(x_*a, y_*a);
	}

	Vector2D<T> operator / (const T& a) const
    {
		T one_over_a = (T)1/a;
		return Vector2D<T>(x_*one_over_a, y_*one_over_a);
	}

	T getMagnitude() const
	{
		return std::sqrt(x_*x_ + y_*y_);
	}

	T SqrMagnitude() const
	{
		return x_*x_ + y_*y_;
	}

	bool compareNonorderly(const Vector2D<T>& to_compare) const		// returns true if both have the same components regardless of the order of components.
	{
		if ((i_ == to_compare.i_ && j_ == to_compare.j_) || (i_ == to_compare.j_ && j_ == to_compare.i_)) return true;
		else return false;
	}

	bool isSqrMagnitudeSmallerThan(const T& sqrmagnitude) const
	{
		if(sqrmagnitude > (x_*x_ + y_*y_)) return true;
		else return false;
	}

	void normalize()
	{
		T magnitude = getMagnitude();

//		assert(magnitude > (T)0);

		if(magnitude != 0)
		{
			T s = 1/magnitude;

			x_ *= s;
			y_ *= s;
		}
	}

	void safeNormalize()
	{
		T magnitude = getMagnitude();

		if(magnitude > 1e-8)
		{
			T s = 1/magnitude;

			x_ *= s;
			y_ *= s;
		}
		else
		{
			x_ = 0;
			y_ = 0;
		}
	}

	Vector2D<T> getNormalized() const
	{
		Vector2D<T> normalized_vector(x_, y_);

		normalized_vector.normalize();

		return normalized_vector;
	}

	Vector2D<T> getSafeNormalized() const
	{
		Vector2D<T> normalized_vector(x_, y_);

		normalized_vector.safeNormalize();

		return normalized_vector;
	}

	void scalingComponents(const Vector2D<T>& normal, const T& normal_coef, const T& tangential_coef)
	{
		const T alpha = dotProduct(*this, normal);
		(*this) -= alpha*normal;// remove normal component and leave tangential component only
		(*this) = (normal_coef*alpha)*normal + tangential_coef*(*this);
	}

	void assign(const T& x_input, const T& y_input)
	{
		x_ = x_input;
		y_ = y_input;
	}

	void assignZeroVector()
	{
		x_ = (T)0;
		y_ = (T)0;
	}

	// this = v1 - v2
	void assignDifference(const Vector2D<T>& v1, const Vector2D<T>& v2)
	{
		x_ = v1.x_ - v2.x_;
		y_ = v1.y_ - v2.y_;
	}

	// this = (p1-p2) + (v1-v2)*dt
	void assignDifferencePlusScaledDifference(const Vector2D<T>& p1, const Vector2D<T>& p2, const Vector2D<T>& v1, const Vector2D<T>& v2, const T& dt)
	{
		x_ = (p1.x_-p2.x_) + (v1.x_-v2.x_)*dt;
		y_ = (p1.y_-p2.y_) + (v1.y_-v2.y_)*dt;
	}

	void assignScaledDifference(const T& scalar, const Vector2D<T>& v1, const Vector2D<T>& v2)
	{
		x_ = scalar*(v1.x_ - v2.x_);
		y_ = scalar*(v1.y_ - v2.y_);
	}

	void assignScaledVector(const T& scalar, const Vector2D<T>& v1)
	{
		x_ = scalar*v1.x_;
		y_ = scalar*v1.y_;
	}

	void add(const T& x_input, const T& y_input)
	{
		x_ += x_input;
		y_ += y_input;
	}

	void addSum(const Vector2D<T>& v1, const Vector2D<T>& v2)
	{
		x_ += (v1.x_ + v2.x_);
		y_ += (v1.y_ + v2.y_);
	}

	void subtractSum(const Vector2D<T>& v1, const Vector2D<T>& v2)
	{
		x_ -= (v1.x_ + v2.x_);
		y_ -= (v1.y_ + v2.y_);
	}

	bool isInside(const T& t)
	{
		if (t < t_min_) return false;
		else if (t > t_max_) return false;
		else return true;
	}

	Vector2D<T> operator - () const
	{
		return Vector2D<T>(-x_, -y_);
	}
};

// miscellaneous free operators and functions

template<class T> bool operator == (const Vector2D<T>& lhs, const Vector2D<T>& rhs)
{
	if (lhs.i_ != rhs.i_) return false;
	else if (lhs.j_ != rhs.j_) return false;
	else return true;
}

template<class T> const Vector2D<T> operator * (const T& a, const Vector2D<T>& v)
{
	return Vector2D<T>(a*v.x_, a*v.y_);
}

template<class T> const T dotProduct(const Vector2D<T>& v1, const Vector2D<T>& v2)
{
	return v1.x_*v2.x_ + v1.y_*v2.y_;
}

template<class T> const T crossProduct(const Vector2D<T>& v1,const Vector2D<T>& v2)
{
	return v1.x_*v2.y_ - v2.x_*v1.y_;
}

template<class T> static bool isSqrDistanceSmallerThan(const Vector2D<T>& a, const Vector2D<T>& b, const T& sqrmagnitude)
{
	const T diff_x(a.x_ - b.x_), diff_y(a.y_ - b.y_);
	if(sqrmagnitude > (diff_x*diff_x + diff_y*diff_y)) return true;
	else return false;
}

template<class T> std::ostream&
operator<<(std::ostream& output,const Vector2D<T>& v)
{
	return output<<v.x_<<" "<<v.y_;
}

// end of file
