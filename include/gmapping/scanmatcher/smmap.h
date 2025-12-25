#ifndef SMMAP_H
#define SMMAP_H
#include <gmapping/grid/map.h>
#include <gmapping/grid/harray2d.h>
#include <gmapping/utils/point.h>
#define SIGHT_INC 1

namespace GMapping {

struct PointAccumulator{
	typedef point<float> FloatPoint;
	/* before
	PointAccumulator(int i=-1): acc(0,0), n(0), visits(0){assert(i==-1);}
	*/
	/*after begin*/
	PointAccumulator(): acc(0,0), n(0), visits(0), intensity_sum(0), intensity_count(0){}
	PointAccumulator(int i): acc(0,0), n(0), visits(0), intensity_sum(0), intensity_count(0){assert(i==-1);}
	/*after end*/
        inline void update(bool value, const Point& p=Point(0,0));
	inline void updateIntensity(float intensity);
	inline Point mean() const {return 1./n*Point(acc.x, acc.y);}
	inline double meanIntensity() const { return intensity_count > 0 ? intensity_sum / intensity_count : 0.0; }
	inline operator double() const { return visits?(double)n*SIGHT_INC/(double)visits:-1; }
	inline void add(const PointAccumulator& p) {
		acc=acc+p.acc; n+=p.n; visits+=p.visits;
		intensity_sum += p.intensity_sum; intensity_count += p.intensity_count;
	}
	static const PointAccumulator& Unknown();
	static PointAccumulator* unknown_ptr;
	FloatPoint acc;
	int n, visits;
	double intensity_sum;      // Accumulated intensity values
	int intensity_count;       // Number of intensity observations
	inline double entropy() const;
};

void PointAccumulator::update(bool value, const Point& p){
	if (value) {
		acc.x+= static_cast<float>(p.x);
		acc.y+= static_cast<float>(p.y);
		n++;
		visits+=SIGHT_INC;
	} else
		visits++;
}

void PointAccumulator::updateIntensity(float intensity){
	if (intensity >= 0) {
		intensity_sum += intensity;
		intensity_count++;
	}
}

double PointAccumulator::entropy() const{
	if (!visits)
		return -log(.5);
	if (n==visits || n==0)
		return 0;
	double x=(double)n*SIGHT_INC/(double)visits;
	return -( x*log(x)+ (1-x)*log(1-x) );
}


typedef Map<PointAccumulator,HierarchicalArray2D<PointAccumulator> > ScanMatcherMap;

};

#endif 
