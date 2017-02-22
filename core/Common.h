#pragma once

#ifndef PI
	#define PI 3.14159265358979323846
#endif

typedef double	D;
typedef float	F;

#define ABS(a) ((a) > 0 ? (a) : -(a))

#define MIN2(a, b)							((a) > (b) ? (b) : (a))
#define MIN3(a, b, c)						(MIN2(MIN2(a, b), (c)))
#define MIN4(a, b, c, d)					(MIN2(MIN3(a, b, c), (d)))
#define MIN5(a, b, c, d, e)					(MIN2(MIN4(a, b, c, d), (e)))
#define MIN7(a, b, c, d, e, f, g)			(MIN2(MIN3(a, b, c), MIN4(d, e, f, g)))
#define MIN8(a, b, c, d, e, f, g, h)		(MIN2(MIN7(a, b, c, d, e, f, g), h))

#define MAX2(a, b)							((a) > (b) ? (a) : (b))
#define MAX3(a, b, c)						(MAX2(MAX2(a, b), (c)))
#define MAX4(a, b, c, d)					(MAX2(MAX3(a, b, c), (d)))
#define MAX5(a, b, c, d, e)					(MAX2(MAX4(a, b, c, d), (e)))
#define MAX7(a, b, c, d, e, f, g)			(MAX2(MAX3(a, b, c), MAX4(d, e, f, g)))
#define MAX8(a, b, c, d, e, f, g, h)		(MAX2(MAX7(a, b, c, d, e, f, g), h))

#define MIN_ABS2(a, b)						(ABS(a) > ABS(b) ? (b) : (a))
#define MAX_ABS2(a, b)						(ABS(a) > ABS(b) ? (a) : (b))

#define CLAMP(v, min, max)		((v) > (max) ? (max) : ((v) < (min) ? (min) : (v)))

#define SQUARE(a)				((a)*(a))
#define POW2(a)					SQUARE(a)
#define POW3(a)					(POW2(a)*a)
#define POW4(a)					(POW3(a)*a)
#define POW5(a)					(POW4(a)*a)
#define POW6(a)					(POW5(a)*a)
#define POW7(a)					(POW6(a)*a)
#define POW8(a)					(POW7(a)*a)

#define SWAP(a,b,type)			{type __temp__ = a; a = b; b = __temp__;}

inline int POW_OF_TWO(const int& n) {
	const int table[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };
	return table[n];
}
inline int POW_OF_EIGHT(const int& n) {
	const int table[] = { 1, 8, 64, 512, 4096 };
	return table[n];
}

//Note: a == 0 returns negative because phi <= 0 means inside.
#define SIGN(a) (a > 0 ? 1 : -1) 

#define SAFE_DELETE(pointer) if(pointer != nullptr){delete pointer; pointer=nullptr;}
#define SAFE_DELETE_ARRAY(pointer) if(pointer != nullptr){delete [] pointer; pointer=nullptr;}

// sort a, b, c -> a1 < a2 < a3	
#define INCREASING_SORT3(a, b, c, a1, a2, a3)		if(a <= b){										\
														if(b <= c){a1 = a;a2 = b;a3 = c;}			\
														else if(a <= c){a1 = a;a2 = c;a3 = b;}		\
														else{a1 = c;a2 = a;a3 = b;}}				\
													 else{											\
														if(a <= c){a1 = b;a2 = a;a3 = c;}			\
														else if(b <= c){a1 = b;a2 = c;a3 = a;}		\
														else{a1 = c;a2 = b;a3 = a;}}

#define INCREASING_SORT2(a, b, a1, a2)				if(a <= b){a1 = a; a2 = b;} \
													else{a1 = b; a2 = a;}													 

// instead of using std::endl which is not good~ 
const char endl [] = "\n";

// end of file
