#ifndef SCANMATCHER_H
#define SCANMATCHER_H

#include "gmapping/scanmatcher/icp.h"
#include "gmapping/scanmatcher/smmap.h"
#include <gmapping/utils/macro_params.h>
#include <gmapping/utils/stat.h>
#include <iostream>
#include <gmapping/utils/gvalues.h>
#include <gmapping/scanmatcher/scanmatcher_export.h>
#define LASER_MAXBEAMS 4096

namespace GMapping {

class SCANMATCHER_EXPORT ScanMatcher{
	public:
		typedef Covariance3 CovarianceMatrix;
		
		ScanMatcher();
		~ScanMatcher();
		double icpOptimize(OrientedPoint& pnew, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		double optimize(OrientedPoint& pnew, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		double optimize(OrientedPoint& mean, CovarianceMatrix& cov, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		
		double   registerScan(ScanMatcherMap& map, const OrientedPoint& p, const double* readings);
		// Register scan using a pre-computed activeArea (for optimization - avoids recomputing ray tracing)
		double   registerScanWithActiveArea(ScanMatcherMap& map, const OrientedPoint& p, const double* readings,
		                                     const HierarchicalArray2D<PointAccumulator>::PointSet& activeArea);
		void setLaserParameters
			(unsigned int beams, double* angles, const OrientedPoint& lpose);
		void setMatchingParameters
			(double urange, double range, double sigma, int kernsize, double lopt, double aopt, int iterations, double likelihoodSigma=1, unsigned int likelihoodSkip=0 );
		void invalidateActiveArea();
		void computeActiveArea(ScanMatcherMap& map, const OrientedPoint& p, const double* readings);
		// Get the last computed activeArea from a map's storage
		static const HierarchicalArray2D<PointAccumulator>::PointSet& getActiveArea(const ScanMatcherMap& map) {
			return map.storage().getActiveArea();
		}

		inline double icpStep(OrientedPoint & pret, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		inline double score(const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		inline unsigned int likelihoodAndScore(double& s, double& l, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const;
		double likelihood(double& lmax, OrientedPoint& mean, CovarianceMatrix& cov, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings);
		double likelihood(double& _lmax, OrientedPoint& _mean, CovarianceMatrix& _cov, const ScanMatcherMap& map, const OrientedPoint& p, Gaussian3& odometry, const double* readings, double gain=180.);
		inline const double* laserAngles() const { return m_laserAngles; }
		inline unsigned int laserBeams() const { return m_laserBeams; }
		
		static const double nullLikelihood;
	protected:
		//state of the matcher
		bool m_activeAreaComputed;
		
		/**laser parameters*/
		unsigned int m_laserBeams;
		double       m_laserAngles[LASER_MAXBEAMS];
		double       m_laserSin[LASER_MAXBEAMS];  // Pre-computed sin(angle) for each beam
		double       m_laserCos[LASER_MAXBEAMS];  // Pre-computed cos(angle) for each beam
		//OrientedPoint m_laserPose;
		PARAM_SET_GET(OrientedPoint, laserPose, protected, public, public)
		PARAM_SET_GET(double, laserMaxRange, protected, public, public)
		/**scan_matcher parameters*/
		PARAM_SET_GET(double, usableRange, protected, public, public)
		PARAM_SET_GET(double, gaussianSigma, protected, public, public)
		PARAM_SET_GET(double, likelihoodSigma, protected, public, public)
		PARAM_SET_GET(int,    kernelSize, protected, public, public)
		PARAM_SET_GET(double, optAngularDelta, protected, public, public)
		PARAM_SET_GET(double, optLinearDelta, protected, public, public)
		PARAM_SET_GET(unsigned int, optRecursiveIterations, protected, public, public)
		PARAM_SET_GET(unsigned int, likelihoodSkip, protected, public, public)
		PARAM_SET_GET(double, llsamplerange, protected, public, public)
		PARAM_SET_GET(double, llsamplestep, protected, public, public)
		PARAM_SET_GET(double, lasamplerange, protected, public, public)
		PARAM_SET_GET(double, lasamplestep, protected, public, public)
		PARAM_SET_GET(bool, generateMap, protected, public, public)
		PARAM_SET_GET(double, enlargeStep, protected, public, public)
		PARAM_SET_GET(double, fullnessThreshold, protected, public, public)
		PARAM_SET_GET(double, angularOdometryReliability, protected, public, public)
		PARAM_SET_GET(double, linearOdometryReliability, protected, public, public)
		PARAM_SET_GET(double, freeCellRatio, protected, public, public)
		PARAM_SET_GET(unsigned int, initialBeamsSkip, protected, public, public)

		// allocate this large array only once
		IntPoint* m_linePoints;
};

inline double ScanMatcher::icpStep(OrientedPoint & pret, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const{
	OrientedPoint lp=p;
	lp.x+=cos(p.theta)*m_laserPose.x-sin(p.theta)*m_laserPose.y;
	lp.y+=sin(p.theta)*m_laserPose.x+cos(p.theta)*m_laserPose.y;
	lp.theta+=m_laserPose.theta;

	// Pre-compute sin/cos of robot orientation (optimization)
	const double cos_theta = cos(lp.theta);
	const double sin_theta = sin(lp.theta);

	unsigned int skip=0;
	double freeDelta=map.getDelta()*m_freeCellRatio;
	std::list<PointPair> pairs;

	for (unsigned int i = m_initialBeamsSkip; i < m_laserBeams; i++){
		const double r = readings[i];
		skip++;
		skip=skip>m_likelihoodSkip?0:skip;
		if (r>m_usableRange||r==0.0) continue;
		if (skip) continue;

		// Use angle addition formula: cos(A+B) = cosA*cosB - sinA*sinB
		const double cos_total = cos_theta * m_laserCos[i] - sin_theta * m_laserSin[i];
		const double sin_total = sin_theta * m_laserCos[i] + cos_theta * m_laserSin[i];

		Point phit=lp;
		phit.x+=r*cos_total;
		phit.y+=r*sin_total;
		IntPoint iphit=map.world2map(phit);
		Point pfree=lp;
		pfree.x+=(r-map.getDelta()*freeDelta)*cos_total;
		pfree.y+=(r-map.getDelta()*freeDelta)*sin_total;
 		pfree=pfree-phit;
		IntPoint ipfree=map.world2map(pfree);
		bool found=false;
		Point bestMu(0.,0.);
		Point bestCell(0.,0.);
		for (int xx=-m_kernelSize; xx<=m_kernelSize; xx++)
		for (int yy=-m_kernelSize; yy<=m_kernelSize; yy++){
			IntPoint pr=iphit+IntPoint(xx,yy);
			IntPoint pf=pr+ipfree;
			//AccessibilityState s=map.storage().cellState(pr);
			//if (s&Inside && s&Allocated){
				const PointAccumulator& cell=map.cell(pr);
				const PointAccumulator& fcell=map.cell(pf);
				if (((double)cell )> m_fullnessThreshold && ((double)fcell )<m_fullnessThreshold){
					Point mu=phit-cell.mean();
					if (!found){
						bestMu=mu;
						bestCell=cell.mean();
						found=true;
					}else
						if((mu*mu)<(bestMu*bestMu)){
							bestMu=mu;
							bestCell=cell.mean();
						}

				}
			//}
		}
		if (found){
			pairs.push_back(std::make_pair(phit, bestCell));
			//std::cerr << "(" << phit.x-bestCell.x << "," << phit.y-bestCell.y << ") ";
		}
		//std::cerr << std::endl;
	}

	OrientedPoint result(0,0,0);
	//double icpError=icpNonlinearStep(result,pairs);
	std::cerr << "result(" << pairs.size() << ")=" << result.x << " " << result.y << " " << result.theta << std::endl;
	pret.x=p.x+result.x;
	pret.y=p.y+result.y;
	pret.theta=p.theta+result.theta;
	pret.theta=atan2(sin(pret.theta), cos(pret.theta));
	return score(map, p, readings);
}

inline double ScanMatcher::score(const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const{
	double s=0;
	OrientedPoint lp=p;
	lp.x+=cos(p.theta)*m_laserPose.x-sin(p.theta)*m_laserPose.y;
	lp.y+=sin(p.theta)*m_laserPose.x+cos(p.theta)*m_laserPose.y;
	lp.theta+=m_laserPose.theta;

	// Pre-compute sin/cos of robot orientation (optimization)
	const double cos_theta = cos(lp.theta);
	const double sin_theta = sin(lp.theta);

	unsigned int skip=0;
	double freeDelta=map.getDelta()*m_freeCellRatio;
	for (unsigned int i = m_initialBeamsSkip; i < m_laserBeams; i++){
		const double r = readings[i];
		skip++;
		skip=skip>m_likelihoodSkip?0:skip;
		if (skip||r>m_usableRange||r==0.0) continue;

		// Use angle addition formula: cos(A+B) = cosA*cosB - sinA*sinB
		const double cos_total = cos_theta * m_laserCos[i] - sin_theta * m_laserSin[i];
		const double sin_total = sin_theta * m_laserCos[i] + cos_theta * m_laserSin[i];

		Point phit=lp;
		phit.x+=r*cos_total;
		phit.y+=r*sin_total;
		IntPoint iphit=map.world2map(phit);
		Point pfree=lp;
		pfree.x+=(r-map.getDelta()*freeDelta)*cos_total;
		pfree.y+=(r-map.getDelta()*freeDelta)*sin_total;
 		pfree=pfree-phit;
		IntPoint ipfree=map.world2map(pfree);
		bool found=false;
		Point bestMu(0.,0.);
		for (int xx=-m_kernelSize; xx<=m_kernelSize; xx++)
		for (int yy=-m_kernelSize; yy<=m_kernelSize; yy++){
			IntPoint pr=iphit+IntPoint(xx,yy);
			IntPoint pf=pr+ipfree;
			//AccessibilityState s=map.storage().cellState(pr);
			//if (s&Inside && s&Allocated){
				const PointAccumulator& cell=map.cell(pr);
				const PointAccumulator& fcell=map.cell(pf);
				if (((double)cell )> m_fullnessThreshold && ((double)fcell )<m_fullnessThreshold){
					Point mu=phit-cell.mean();
					if (!found){
						bestMu=mu;
						found=true;
					}else
						bestMu=(mu*mu)<(bestMu*bestMu)?mu:bestMu;
				}
			//}
		}
		if (found)
			s+=exp(-1./m_gaussianSigma*bestMu*bestMu);
	}
	return s;
}

inline unsigned int ScanMatcher::likelihoodAndScore(double& s, double& l, const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const{
	using namespace std;
	l=0;
	s=0;
	OrientedPoint lp=p;
	lp.x+=cos(p.theta)*m_laserPose.x-sin(p.theta)*m_laserPose.y;
	lp.y+=sin(p.theta)*m_laserPose.x+cos(p.theta)*m_laserPose.y;
	lp.theta+=m_laserPose.theta;

	// Pre-compute sin/cos of robot orientation (optimization)
	const double cos_theta = cos(lp.theta);
	const double sin_theta = sin(lp.theta);

	double noHit=nullLikelihood/(m_likelihoodSigma);
	unsigned int skip=0;
	unsigned int c=0;
	double freeDelta=map.getDelta()*m_freeCellRatio;
	for (unsigned int i = m_initialBeamsSkip; i < m_laserBeams; i++){
		const double r = readings[i];
		skip++;
		skip=skip>m_likelihoodSkip?0:skip;
		if (r>m_usableRange) continue;
		if (skip) continue;

		// Use angle addition formula: cos(A+B) = cosA*cosB - sinA*sinB
		const double cos_total = cos_theta * m_laserCos[i] - sin_theta * m_laserSin[i];
		const double sin_total = sin_theta * m_laserCos[i] + cos_theta * m_laserSin[i];

		Point phit=lp;
		phit.x+=r*cos_total;
		phit.y+=r*sin_total;
		IntPoint iphit=map.world2map(phit);
		Point pfree=lp;
		pfree.x+=(r-freeDelta)*cos_total;
		pfree.y+=(r-freeDelta)*sin_total;
		pfree=pfree-phit;
		IntPoint ipfree=map.world2map(pfree);
		bool found=false;
		Point bestMu(0.,0.);
		for (int xx=-m_kernelSize; xx<=m_kernelSize; xx++)
		for (int yy=-m_kernelSize; yy<=m_kernelSize; yy++){
			IntPoint pr=iphit+IntPoint(xx,yy);
			IntPoint pf=pr+ipfree;
			//AccessibilityState s=map.storage().cellState(pr);
			//if (s&Inside && s&Allocated){
				const PointAccumulator& cell=map.cell(pr);
				const PointAccumulator& fcell=map.cell(pf);
				if (((double)cell )>m_fullnessThreshold && ((double)fcell )<m_fullnessThreshold){
					Point mu=phit-cell.mean();
					if (!found){
						bestMu=mu;
						found=true;
					}else
						bestMu=(mu*mu)<(bestMu*bestMu)?mu:bestMu;
				}
			//}
		}
		if (found){
			s+=exp(-1./m_gaussianSigma*bestMu*bestMu);
			c++;
		}
		if (!skip){
			double f=(-1./m_likelihoodSigma)*(bestMu*bestMu);
			l+=(found)?f:noHit;
		}
	}
	return c;
}

};

#endif
