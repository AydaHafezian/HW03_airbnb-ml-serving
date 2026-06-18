# HW03 Docker Image Size Report

| Repository | Tag | Size |
|---|---|---|
| qbc12-airbnb-serving | optimized | 1.31GB |
| qbc12-airbnb-serving | naive | 3.15GB |

## Analysis
The optimized image is much smaller than the naive image because it uses a multi-stage build.
In the optimized version, build tools and intermediate dependencies are only used in the builder stage and are not included in the final runtime image.
For production, I would use the optimized image because it is smaller, faster to deploy, and has a lower attack surface.
