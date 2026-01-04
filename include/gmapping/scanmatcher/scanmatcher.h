#ifndef SCANMATCHER_H
#define SCANMATCHER_H

#include "gmapping/scanmatcher/icp.h"
#include "gmapping/scanmatcher/smmap.h"
#include <gmapping/utils/macro_params.h>
#include <gmapping/utils/stat.h>
#include <iostream>
#include <gmapping/utils/gvalues.h>
#include <gmapping/scanmatcher/scanmatcher_export.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

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
		// Float versions for NEON optimization (aligned for SIMD)
		alignas(16) float m_laserSinF[LASER_MAXBEAMS];
		alignas(16) float m_laserCosF[LASER_MAXBEAMS];
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

#ifndef __ARM_NEON
// Standard (non-NEON) score function
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
#endif // !__ARM_NEON

#ifdef __ARM_NEON
/**
 * NEON-optimized score function for ARM processors
 * Uses float32x4 to process 4 beams in parallel where possible
 * Falls back to scalar for map lookups (which can't be vectorized)
 */
inline double ScanMatcher::score(const ScanMatcherMap& map, const OrientedPoint& p, const double* readings) const{
	double s=0;
	OrientedPoint lp=p;
	lp.x+=cos(p.theta)*m_laserPose.x-sin(p.theta)*m_laserPose.y;
	lp.y+=sin(p.theta)*m_laserPose.x+cos(p.theta)*m_laserPose.y;
	lp.theta+=m_laserPose.theta;

	// Use float for faster computation on ARM
	const float cos_theta = static_cast<float>(cos(lp.theta));
	const float sin_theta = static_cast<float>(sin(lp.theta));
	const float lp_x = static_cast<float>(lp.x);
	const float lp_y = static_cast<float>(lp.y);
	const float freeDelta = static_cast<float>(map.getDelta()*m_freeCellRatio);
	const float invSigma = static_cast<float>(-1.0/m_gaussianSigma);

	// NEON vectors for batch processing
	const float32x4_t cos_t = vdupq_n_f32(cos_theta);
	const float32x4_t sin_t = vdupq_n_f32(sin_theta);
	const float32x4_t lpx = vdupq_n_f32(lp_x);
	const float32x4_t lpy = vdupq_n_f32(lp_y);

	unsigned int skip=0;
	unsigned int i = m_initialBeamsSkip;

	// Aligned buffers for NEON
	alignas(16) float r4[4];
	alignas(16) float hit_x[4], hit_y[4];
	alignas(16) float free_dx[4], free_dy[4];

	// Process 4 beams at a time
	while (i + 4 <= m_laserBeams) {
		int validCount = 0;
		int validIdx[4];

		// Find next 4 valid beams
		for (int batch = 0; batch < 4 && i < m_laserBeams; ) {
			skip++;
			if (skip > m_likelihoodSkip) skip = 0;

			float r = static_cast<float>(readings[i]);
			if (!skip && r <= m_usableRange && r > 0.0f) {
				r4[validCount] = r;
				validIdx[validCount] = i;
				validCount++;
				batch++;
			}
			i++;
		}

		if (validCount == 0) continue;

		// Load sin/cos for valid beams
		alignas(16) float lcos[4] = {0}, lsin[4] = {0};
		for (int j = 0; j < validCount; j++) {
			lcos[j] = m_laserCosF[validIdx[j]];
			lsin[j] = m_laserSinF[validIdx[j]];
		}

		// NEON: compute cos_total and sin_total for 4 beams
		float32x4_t vlcos = vld1q_f32(lcos);
		float32x4_t vlsin = vld1q_f32(lsin);
		float32x4_t vr = vld1q_f32(r4);

		// cos_total = cos_theta * laserCos - sin_theta * laserSin
		float32x4_t cos_total = vsubq_f32(
			vmulq_f32(cos_t, vlcos),
			vmulq_f32(sin_t, vlsin)
		);

		// sin_total = sin_theta * laserCos + cos_theta * laserSin
		float32x4_t sin_total = vaddq_f32(
			vmulq_f32(sin_t, vlcos),
			vmulq_f32(cos_t, vlsin)
		);

		// hit = lp + r * (cos_total, sin_total)
		float32x4_t hx = vaddq_f32(lpx, vmulq_f32(vr, cos_total));
		float32x4_t hy = vaddq_f32(lpy, vmulq_f32(vr, sin_total));

		// free_delta = (r - freeDelta) * (cos_total, sin_total) - hit
		float32x4_t vfreeDelta = vdupq_n_f32(freeDelta);
		float32x4_t r_minus_fd = vsubq_f32(vr, vfreeDelta);
		float32x4_t fx = vsubq_f32(vmulq_f32(r_minus_fd, cos_total), vmulq_f32(vr, cos_total));
		float32x4_t fy = vsubq_f32(vmulq_f32(r_minus_fd, sin_total), vmulq_f32(vr, sin_total));

		vst1q_f32(hit_x, hx);
		vst1q_f32(hit_y, hy);
		vst1q_f32(free_dx, fx);
		vst1q_f32(free_dy, fy);

		// Process each valid beam (map lookup can't be vectorized)
		for (int j = 0; j < validCount; j++) {
			Point phit(hit_x[j], hit_y[j]);
			IntPoint iphit = map.world2map(phit);
			IntPoint ipfree(
				static_cast<int>(free_dx[j] / map.getDelta()),
				static_cast<int>(free_dy[j] / map.getDelta())
			);

			bool found = false;
			float bestMuSq = 0;

			for (int xx = -m_kernelSize; xx <= m_kernelSize; xx++) {
				for (int yy = -m_kernelSize; yy <= m_kernelSize; yy++) {
					IntPoint pr = iphit + IntPoint(xx, yy);
					IntPoint pf = pr + ipfree;

					const PointAccumulator& cell = map.cell(pr);
					const PointAccumulator& fcell = map.cell(pf);

					if (((double)cell) > m_fullnessThreshold && ((double)fcell) < m_fullnessThreshold) {
						Point mu = phit - cell.mean();
						float muSq = static_cast<float>(mu.x*mu.x + mu.y*mu.y);

						if (!found || muSq < bestMuSq) {
							bestMuSq = muSq;
							found = true;
						}
					}
				}
			}

			if (found) {
				s += exp(invSigma * bestMuSq);
			}
		}
	}

	// Process remaining beams scalar
	for (; i < m_laserBeams; i++) {
		skip++;
		if (skip > m_likelihoodSkip) skip = 0;

		float r = static_cast<float>(readings[i]);
		if (skip || r > m_usableRange || r <= 0.0f) continue;

		float cos_total = cos_theta * m_laserCosF[i] - sin_theta * m_laserSinF[i];
		float sin_total = sin_theta * m_laserCosF[i] + cos_theta * m_laserSinF[i];

		Point phit(lp_x + r * cos_total, lp_y + r * sin_total);
		IntPoint iphit = map.world2map(phit);

		float free_dx_s = (r - freeDelta) * cos_total - r * cos_total;
		float free_dy_s = (r - freeDelta) * sin_total - r * sin_total;
		IntPoint ipfree(
			static_cast<int>(free_dx_s / map.getDelta()),
			static_cast<int>(free_dy_s / map.getDelta())
		);

		bool found = false;
		float bestMuSq = 0;

		for (int xx = -m_kernelSize; xx <= m_kernelSize; xx++) {
			for (int yy = -m_kernelSize; yy <= m_kernelSize; yy++) {
				IntPoint pr = iphit + IntPoint(xx, yy);
				IntPoint pf = pr + ipfree;

				const PointAccumulator& cell = map.cell(pr);
				const PointAccumulator& fcell = map.cell(pf);

				if (((double)cell) > m_fullnessThreshold && ((double)fcell) < m_fullnessThreshold) {
					Point mu = phit - cell.mean();
					float muSq = static_cast<float>(mu.x*mu.x + mu.y*mu.y);

					if (!found || muSq < bestMuSq) {
						bestMuSq = muSq;
						found = true;
					}
				}
			}
		}

		if (found) {
			s += exp(invSigma * bestMuSq);
		}
	}

	return s;
}
#endif // __ARM_NEON

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
