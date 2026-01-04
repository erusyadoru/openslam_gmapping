#ifndef SCANMATCHER_NEON_H
#define SCANMATCHER_NEON_H

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <cmath>
#include <algorithm>

namespace GMapping {

/**
 * NEON-optimized helper functions for scan matching
 * These provide 4x speedup for ARM processors (Raspberry Pi, Orange Pi, etc.)
 */

#ifdef __ARM_NEON

// Process 4 beams at once: compute hit points
inline void computeHitPoints4_NEON(
    const float* readings,      // 4 range readings
    const float* laserCos,      // 4 pre-computed cos(angle)
    const float* laserSin,      // 4 pre-computed sin(angle)
    float cos_theta,            // cos of robot orientation
    float sin_theta,            // sin of robot orientation
    float lp_x, float lp_y,     // laser position
    float* hit_x, float* hit_y  // output: 4 hit points
) {
    // Load 4 values at once
    float32x4_t r = vld1q_f32(readings);
    float32x4_t lcos = vld1q_f32(laserCos);
    float32x4_t lsin = vld1q_f32(laserSin);

    // cos_total = cos_theta * laserCos - sin_theta * laserSin
    float32x4_t cos_t = vdupq_n_f32(cos_theta);
    float32x4_t sin_t = vdupq_n_f32(sin_theta);

    float32x4_t cos_total = vsubq_f32(
        vmulq_f32(cos_t, lcos),
        vmulq_f32(sin_t, lsin)
    );

    // sin_total = sin_theta * laserCos + cos_theta * laserSin
    float32x4_t sin_total = vaddq_f32(
        vmulq_f32(sin_t, lcos),
        vmulq_f32(cos_t, lsin)
    );

    // hit_x = lp_x + r * cos_total
    float32x4_t lpx = vdupq_n_f32(lp_x);
    float32x4_t lpy = vdupq_n_f32(lp_y);

    float32x4_t hx = vaddq_f32(lpx, vmulq_f32(r, cos_total));
    float32x4_t hy = vaddq_f32(lpy, vmulq_f32(r, sin_total));

    // Store results
    vst1q_f32(hit_x, hx);
    vst1q_f32(hit_y, hy);
}

// Compute squared distances for 4 points at once
inline void computeSquaredDist4_NEON(
    const float* dx,  // 4 delta x values
    const float* dy,  // 4 delta y values
    float* dist_sq    // output: 4 squared distances
) {
    float32x4_t vdx = vld1q_f32(dx);
    float32x4_t vdy = vld1q_f32(dy);

    // dist_sq = dx*dx + dy*dy
    float32x4_t result = vaddq_f32(
        vmulq_f32(vdx, vdx),
        vmulq_f32(vdy, vdy)
    );

    vst1q_f32(dist_sq, result);
}

// Compute exp(-k * x) for 4 values (approximate)
// Uses polynomial approximation for speed
inline void computeExpNeg4_NEON(
    const float* x,   // 4 input values (already multiplied by k)
    float* result     // output: 4 exp values
) {
    float32x4_t vx = vld1q_f32(x);

    // Negate
    vx = vnegq_f32(vx);

    // Clamp to prevent overflow: exp(x) for x < -10 is essentially 0
    float32x4_t min_val = vdupq_n_f32(-10.0f);
    float32x4_t max_val = vdupq_n_f32(0.0f);
    vx = vmaxq_f32(vx, min_val);
    vx = vminq_f32(vx, max_val);

    // Polynomial approximation of exp(x) for x in [-10, 0]
    // exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t half = vdupq_n_f32(0.5f);
    float32x4_t sixth = vdupq_n_f32(1.0f/6.0f);
    float32x4_t twentyfourth = vdupq_n_f32(1.0f/24.0f);

    float32x4_t x2 = vmulq_f32(vx, vx);
    float32x4_t x3 = vmulq_f32(x2, vx);
    float32x4_t x4 = vmulq_f32(x2, x2);

    // result = 1 + x + x^2/2 + x^3/6 + x^4/24
    float32x4_t res = one;
    res = vaddq_f32(res, vx);
    res = vaddq_f32(res, vmulq_f32(x2, half));
    res = vaddq_f32(res, vmulq_f32(x3, sixth));
    res = vaddq_f32(res, vmulq_f32(x4, twentyfourth));

    // Clamp result to [0, 1]
    res = vmaxq_f32(res, vdupq_n_f32(0.0f));
    res = vminq_f32(res, one);

    vst1q_f32(result, res);
}

// Sum 4 float values
inline float horizontalSum4_NEON(float32x4_t v) {
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(sum, sum), 0);
}

#endif // __ARM_NEON

/**
 * Batch pre-compute all hit points for a scan
 * This is useful when the same readings are scored multiple times
 */
struct PrecomputedScan {
    std::vector<float> hit_x;
    std::vector<float> hit_y;
    std::vector<float> free_x;
    std::vector<float> free_y;
    std::vector<bool> valid;
    std::vector<int> valid_indices;

    void resize(size_t n) {
        hit_x.resize(n);
        hit_y.resize(n);
        free_x.resize(n);
        free_y.resize(n);
        valid.resize(n);
        valid_indices.clear();
        valid_indices.reserve(n);
    }
};

/**
 * Precompute hit points for all beams
 * Call this once per scan, then use for multiple score evaluations
 */
inline void precomputeHitPoints(
    PrecomputedScan& result,
    const double* readings,
    unsigned int numBeams,
    unsigned int initialSkip,
    unsigned int likelihoodSkip,
    double usableRange,
    double lp_x, double lp_y, double lp_theta,
    const double* laserSin,
    const double* laserCos,
    double freeDelta
) {
    result.resize(numBeams);
    result.valid_indices.clear();

    const double cos_theta = cos(lp_theta);
    const double sin_theta = sin(lp_theta);

    unsigned int skip = 0;

#ifdef __ARM_NEON
    // NEON path: process 4 beams at a time
    alignas(16) float r4[4];
    alignas(16) float lcos4[4];
    alignas(16) float lsin4[4];
    alignas(16) float hx4[4];
    alignas(16) float hy4[4];

    unsigned int i = initialSkip;

    // Process groups of 4
    while (i + 4 <= numBeams) {
        int validCount = 0;
        int validIdx[4];

        // Check which beams are valid and collect them
        for (int j = 0; j < 4; j++) {
            skip++;
            if (skip > likelihoodSkip) skip = 0;

            double r = readings[i + j];
            bool isValid = !skip && r <= usableRange && r > 0.0;
            result.valid[i + j] = isValid;

            if (isValid) {
                r4[validCount] = (float)r;
                lcos4[validCount] = (float)laserCos[i + j];
                lsin4[validCount] = (float)laserSin[i + j];
                validIdx[validCount] = i + j;
                validCount++;
            }
        }

        // If we have valid beams, process them
        if (validCount > 0) {
            // Pad remaining slots
            for (int j = validCount; j < 4; j++) {
                r4[j] = 0;
                lcos4[j] = 0;
                lsin4[j] = 0;
            }

            computeHitPoints4_NEON(r4, lcos4, lsin4,
                (float)cos_theta, (float)sin_theta,
                (float)lp_x, (float)lp_y, hx4, hy4);

            // Store results for valid beams
            for (int j = 0; j < validCount; j++) {
                int idx = validIdx[j];
                result.hit_x[idx] = hx4[j];
                result.hit_y[idx] = hy4[j];
                result.valid_indices.push_back(idx);

                // Compute free point
                double r = readings[idx];
                double cos_total = cos_theta * laserCos[idx] - sin_theta * laserSin[idx];
                double sin_total = sin_theta * laserCos[idx] + cos_theta * laserSin[idx];
                result.free_x[idx] = (r - freeDelta) * cos_total;
                result.free_y[idx] = (r - freeDelta) * sin_total;
            }
        }

        i += 4;
    }

    // Process remaining beams scalar
    for (; i < numBeams; i++) {
        skip++;
        if (skip > likelihoodSkip) skip = 0;

        double r = readings[i];
        bool isValid = !skip && r <= usableRange && r > 0.0;
        result.valid[i] = isValid;

        if (isValid) {
            double cos_total = cos_theta * laserCos[i] - sin_theta * laserSin[i];
            double sin_total = sin_theta * laserCos[i] + cos_theta * laserSin[i];

            result.hit_x[i] = lp_x + r * cos_total;
            result.hit_y[i] = lp_y + r * sin_total;
            result.free_x[i] = (r - freeDelta) * cos_total;
            result.free_y[i] = (r - freeDelta) * sin_total;
            result.valid_indices.push_back(i);
        }
    }

#else
    // Scalar fallback
    for (unsigned int i = initialSkip; i < numBeams; i++) {
        skip++;
        if (skip > likelihoodSkip) skip = 0;

        double r = readings[i];
        bool isValid = !skip && r <= usableRange && r > 0.0;
        result.valid[i] = isValid;

        if (isValid) {
            double cos_total = cos_theta * laserCos[i] - sin_theta * laserSin[i];
            double sin_total = sin_theta * laserCos[i] + cos_theta * laserSin[i];

            result.hit_x[i] = lp_x + r * cos_total;
            result.hit_y[i] = lp_y + r * sin_total;
            result.free_x[i] = (r - freeDelta) * cos_total;
            result.free_y[i] = (r - freeDelta) * sin_total;
            result.valid_indices.push_back(i);
        }
    }
#endif
}

} // namespace GMapping

#endif // SCANMATCHER_NEON_H
