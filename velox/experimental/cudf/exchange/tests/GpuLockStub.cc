/*
 * Stub implementations of gluten::lockGpu/unlockGpu for standalone Velox tests.
 * In production, these are provided by the Gluten JNI layer (Java semaphore).
 * For MPP prototype tests, GPU locking is disabled — stubs are no-ops.
 */
namespace gluten {
void lockGpu() {}
void unlockGpu() {}
} // namespace gluten
