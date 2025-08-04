#include "Header.cuh"

// *************************************************************************************************

#define USE_INSERTION_SORT

// *************************************************************************************************

extern "C" __constant__ SLaunchParams optixLaunchParams;

// *************************************************************************************************

const int BUFFER_SIZE = 1024;

// *************************************************************************************************

struct SRayPayload {
	float2 data[BUFFER_SIZE]; // the buffer size has to have more elements than the number of "nearest" neighbors to be found K.
	int neighbors_num;
	float max_dist_so_far;
	float max_dist_so_far_old;
	bool overflow;
};

// *************************************************************************************************

extern "C" __global__ void __raygen__() {
	int x = optixGetLaunchIndex().x;

	float4 queried_point = optixLaunchParams.queried_points[x];
	int number_of_queried_points = optixGetLaunchDimensions().x;

	float3 v = make_float3(1.0f, 0.0f, 0.0f);

	// *** *** *** *** ***

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *** *** *** *** ***

	#ifdef USE_INSERTION_SORT
		// INSERTION SORT
		rp.max_dist_so_far_old = optixLaunchParams.max_R;
		do {
			rp.neighbors_num = 0;
			rp.max_dist_so_far = -INFINITY;
			rp.overflow = false;
		
			optixTrace(
				optixLaunchParams.traversable,
				make_float3(queried_point.x, queried_point.y, queried_point.z),
				v,
				0.0f,
				optixLaunchParams.max_R + rp.max_dist_so_far_old, // !!! !!! !!!
				0.0f,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
				0,
				1,
				0,

				rp_addr_lo,
				rp_addr_hi
			);

			for (int i = 1; i < rp.neighbors_num; ++i) {
				float2 tmp1 = rp.data[i];
			
				int j = (i <= optixLaunchParams.K) ? i : optixLaunchParams.K;
				for (; j > 0; --j) {
					float2 tmp2 = rp.data[j - 1];
				
					if (tmp1.x < tmp2.x) rp.data[j] = tmp2;
					else
						break;
				}
				if (j < i) rp.data[j] = tmp1;
			}

			// !!! !!! !!!
			if (rp.overflow) rp.max_dist_so_far_old = rp.data[optixLaunchParams.K - 1].x;
			// !!! !!! !!!
		} while (rp.overflow);
	#else
		// BUBBLE SORT
		rp.neighbors_num = 0;
	
		optixTrace(
			optixLaunchParams.traversable,
			make_float3(queried_point.x, queried_point.y, queried_point.z),
			v,
			0.0f,
			2.0f * optixLaunchParams.max_R, // !!! !!! !!!
			0.0f,
			OptixVisibilityMask(255),
			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
			0,
			1,
			0,

			rp_addr_lo,
			rp_addr_hi
		);
	#endif

	// *** *** *** *** ***

	for (int i = 0; i < optixLaunchParams.K; ++i) {
		if (i < rp.neighbors_num) {
			float2 tmp = rp.data[i];

			optixLaunchParams.distances[(i * number_of_queried_points) + x] = tmp.x;
			optixLaunchParams.indices[(i * number_of_queried_points) + x] = __float_as_uint(tmp.y);
		} else {
			optixLaunchParams.distances[(i * number_of_queried_points) + x] = -INFINITY;
			optixLaunchParams.indices[(i * number_of_queried_points) + x] = -1;
		}
	}
}

// *************************************************************************************************

extern "C" __global__ void __anyhit__() {
	unsigned gauss_ind = optixGetInstanceIndex();

	float4 mean = optixLaunchParams.means[gauss_ind];
	
	// *** *** *** *** ***

	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *** *** *** *** ***

	#ifdef USE_INSERTION_SORT
		// INSERTION SORT
		float tMin = optixGetRayTmax();
		float3 O = optixGetWorldRayOrigin();
		float3 d = make_float3(mean.x - O.x, mean.y - O.y, mean.z - O.z);
	
		float distance_squared = __fmaf_rn(d.x, d.x, __fmaf_rn(d.y, d.y, d.z * d.z));
		float distance;
		asm volatile (
			"sqrt.approx.ftz.f32 %0, %1;" :
			"=f"(distance) :
			"f"(distance_squared)
		);

		float max_distance;
		asm volatile (
			"sqrt.approx.ftz.f32 %0, %1;" :
			"=f"(max_distance) :
			"f"(optixLaunchParams.chi_square_squared_radius)
		);
		max_distance *= mean.w;

		if ((distance <= max_distance) && (distance <= rp->max_dist_so_far_old)) {
			if (rp->neighbors_num < optixLaunchParams.K) {
				if (distance > rp->max_dist_so_far)	rp->max_dist_so_far = distance;
				rp->data[rp->neighbors_num++] = make_float2(distance, __uint_as_float(gauss_ind));
			} else {
				if (distance <= rp->max_dist_so_far) {
					if (rp->neighbors_num < BUFFER_SIZE) rp->data[rp->neighbors_num++] = make_float2(distance, __uint_as_float(gauss_ind));
					else
						rp->overflow = true;
				}
			}
		}
		if ((rp->neighbors_num < optixLaunchParams.K) || (tMin <= optixLaunchParams.max_R + rp->max_dist_so_far)) optixIgnoreIntersection();
	#else
		// BUBBLE SORT
		float tHit = optixGetRayTmax();
		int neighbors_num = rp->neighbors_num;

		float3 O = optixGetWorldRayOrigin();
		float3 d = make_float3(mean.x - O.x, mean.y - O.y, mean.z - O.z);
		float distance_squared = __fmaf_rn(d.x, d.x, __fmaf_rn(d.y, d.y, d.z * d.z));
		float distance;
		asm volatile (
			"sqrt.approx.ftz.f32 %0, %1;" :
			"=f"(distance) :
			"f"(distance_squared)
		);

		float max_distance;
		asm volatile (
			"sqrt.approx.ftz.f32 %0, %1;" :
			"=f"(max_distance) :
			"f"(optixLaunchParams.chi_square_squared_radius)
		);
		max_distance *= mean.w;

		if (distance <= max_distance) {
			float2 tmp1 = make_float2(distance, __uint_as_float(gauss_ind));
			float2 tmp2;

			for (int i = 0; i < neighbors_num; ++i) {
				tmp2 = rp->data[i];

				if (tmp1.x < tmp2.x) {
					rp->data[i] = tmp1;
					tmp1 = tmp2;
				}
			}

			if (neighbors_num < optixLaunchParams.K) {
				rp->data[neighbors_num++] = tmp1;
				rp->neighbors_num = neighbors_num;

				optixIgnoreIntersection();
			} else {
				if (tHit <= optixLaunchParams.max_R + tmp2.x) optixIgnoreIntersection();
			}
		} else {
			if (neighbors_num < optixLaunchParams.K)
				optixIgnoreIntersection();
			else {
				if (tHit <= optixLaunchParams.max_R + rp->data[optixLaunchParams.K - 1].x) optixIgnoreIntersection();
			}
		}
	#endif
}