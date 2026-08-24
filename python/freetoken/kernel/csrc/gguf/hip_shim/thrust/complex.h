// Minimal thrust::complex stand-in for ROCm builds of torch extensions.
//
// torch's own headers (torch/headeronly/util/complex.h, c10/util/complex_math.h)
// include <thrust/complex.h> on HIP merely to interoperate with c10::complex --
// they need the type, real()/imag(), conversions, thrust::polar() and the
// transcendental set. The full rocThrust header stack cannot be satisfied by
// the pip-bundled SDK (it wants libcu++/hipstd device-stdlib headers that ROCm
// wheels don't ship, and its fallback trait definitions collide under AMD
// clang), so we provide exactly the surface c10 uses. Put this directory FIRST
// on the include path; a real thrust install would shadow it harmlessly.
#pragma once

#include <cmath>

#if defined(__HIP__) || defined(__CUDACC__) || defined(__HIPCC__)
#define FT_THRUST_SHIM_HOST_DEVICE __host__ __device__
#else
#define FT_THRUST_SHIM_HOST_DEVICE
#endif

namespace ft_shim {

// Explicit scalar wrappers over the C-suffix libm/HIP-device builtins.
// Unqualified names like ::std::exp are unavailable in the HIP device pass
// (<cmath> is host-only there), while bare exp/log/sqrt collide with our own
// complex overloads below -- hence this neutral overload set.
#define FT_THRUST_SHIM_SCALAR_FN1(name)                                     \
  FT_THRUST_SHIM_HOST_DEVICE inline float name(float x) { return name##f(x); } \
  FT_THRUST_SHIM_HOST_DEVICE inline double name(double x) { return name(x); }
#define FT_THRUST_SHIM_SCALAR_FN2(name)                                       \
  FT_THRUST_SHIM_HOST_DEVICE inline float name(float x, float y) {            \
    return name##f(x, y);                                                     \
  }                                                                           \
  FT_THRUST_SHIM_HOST_DEVICE inline double name(double x, double y) {         \
    return name(x, y);                                                        \
  }

FT_THRUST_SHIM_HOST_DEVICE inline float abs(float x) { return fabsf(x); }
FT_THRUST_SHIM_HOST_DEVICE inline double abs(double x) { return fabs(x); }
FT_THRUST_SHIM_SCALAR_FN1(acos)
FT_THRUST_SHIM_SCALAR_FN1(acosh)
FT_THRUST_SHIM_SCALAR_FN1(asin)
FT_THRUST_SHIM_SCALAR_FN1(asinh)
FT_THRUST_SHIM_SCALAR_FN1(atan)
FT_THRUST_SHIM_SCALAR_FN1(atanh)
FT_THRUST_SHIM_SCALAR_FN1(cos)
FT_THRUST_SHIM_SCALAR_FN1(cosh)
FT_THRUST_SHIM_SCALAR_FN1(exp)
FT_THRUST_SHIM_SCALAR_FN1(log)
FT_THRUST_SHIM_SCALAR_FN1(log10)
FT_THRUST_SHIM_SCALAR_FN1(sin)
FT_THRUST_SHIM_SCALAR_FN1(sinh)
FT_THRUST_SHIM_SCALAR_FN1(sqrt)
FT_THRUST_SHIM_SCALAR_FN1(tan)
FT_THRUST_SHIM_SCALAR_FN1(tanh)
FT_THRUST_SHIM_SCALAR_FN2(atan2)
FT_THRUST_SHIM_SCALAR_FN2(pow)

#undef FT_THRUST_SHIM_SCALAR_FN1
#undef FT_THRUST_SHIM_SCALAR_FN2

}  // namespace ft_shim

namespace thrust {

// Scalar math lives in ft_shim; make those overloads visible here so
// dependent calls (e.g. log(abs(z)) with z a complex<T>) can resolve them
// during template instantiation.
using namespace ::ft_shim;

template <typename T>
class complex {
 public:
  FT_THRUST_SHIM_HOST_DEVICE complex() : re_(T()), im_(T()) {}
  FT_THRUST_SHIM_HOST_DEVICE complex(const T& re, const T& im = T())
      : re_(re), im_(im) {}

  template <typename U>
  FT_THRUST_SHIM_HOST_DEVICE complex(const complex<U>& other)
      : re_(static_cast<T>(other.real())), im_(static_cast<T>(other.imag())) {}

  FT_THRUST_SHIM_HOST_DEVICE T real() const { return re_; }
  FT_THRUST_SHIM_HOST_DEVICE T imag() const { return im_; }
  FT_THRUST_SHIM_HOST_DEVICE void real(const T& re) { re_ = re; }
  FT_THRUST_SHIM_HOST_DEVICE void imag(const T& im) { im_ = im; }

  FT_THRUST_SHIM_HOST_DEVICE complex<T>& operator=(const T& re) {
    re_ = re;
    im_ = T();
    return *this;
  }

  template <typename U>
  FT_THRUST_SHIM_HOST_DEVICE complex<T>& operator+=(const complex<U>& o) {
    re_ += static_cast<T>(o.real());
    im_ += static_cast<T>(o.imag());
    return *this;
  }
  template <typename U>
  FT_THRUST_SHIM_HOST_DEVICE complex<T>& operator-=(const complex<U>& o) {
    re_ -= static_cast<T>(o.real());
    im_ -= static_cast<T>(o.imag());
    return *this;
  }
  template <typename U>
  FT_THRUST_SHIM_HOST_DEVICE complex<T>& operator*=(const complex<U>& o) {
    const T re = re_ * static_cast<T>(o.real()) - im_ * static_cast<T>(o.imag());
    const T im = re_ * static_cast<T>(o.imag()) + im_ * static_cast<T>(o.real());
    re_ = re;
    im_ = im;
    return *this;
  }
  template <typename U>
  FT_THRUST_SHIM_HOST_DEVICE complex<T>& operator/=(const complex<U>& o) {
    const T denom = static_cast<T>(o.real()) * static_cast<T>(o.real()) +
                    static_cast<T>(o.imag()) * static_cast<T>(o.imag());
    const T re =
        (re_ * static_cast<T>(o.real()) + im_ * static_cast<T>(o.imag())) / denom;
    const T im =
        (im_ * static_cast<T>(o.real()) - re_ * static_cast<T>(o.imag())) / denom;
    re_ = re;
    im_ = im;
    return *this;
  }

 private:
  T re_;
  T im_;
};

template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> operator+(
    const complex<T>& a, const complex<T>& b) {
  return complex<T>(a) += b;
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> operator-(
    const complex<T>& a, const complex<T>& b) {
  return complex<T>(a) -= b;
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> operator*(
    const complex<T>& a, const complex<T>& b) {
  return complex<T>(a) *= b;
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> operator/(
    const complex<T>& a, const complex<T>& b) {
  return complex<T>(a) /= b;
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> operator-(const complex<T>& a) {
  return complex<T>(-a.real(), -a.imag());
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE bool operator==(
    const complex<T>& a, const complex<T>& b) {
  return a.real() == b.real() && a.imag() == b.imag();
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE bool operator!=(
    const complex<T>& a, const complex<T>& b) {
  return !(a == b);
}

template <typename T>
FT_THRUST_SHIM_HOST_DEVICE T abs(const complex<T>& z) {
  return ::ft_shim::sqrt(z.real() * z.real() + z.imag() * z.imag());
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE T arg(const complex<T>& z) {
  return ::ft_shim::atan2(z.imag(), z.real());
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE T norm(const complex<T>& z) {
  return z.real() * z.real() + z.imag() * z.imag();
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> conj(const complex<T>& z) {
  return complex<T>(z.real(), -z.imag());
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> polar(const T& r, const T& theta = T()) {
  return complex<T>(r * ::ft_shim::cos(theta), r * ::ft_shim::sin(theta));
}

template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> exp(const complex<T>& z) {
  return polar(::ft_shim::exp(z.real()), z.imag());
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> log(const complex<T>& z) {
  return complex<T>(::ft_shim::log(abs(z)), arg(z));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> log10(const complex<T>& z) {
  return log(z) / complex<T>(::ft_shim::log(static_cast<T>(10)));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> sqrt(const complex<T>& z) {
  const T r = abs(z);
  if (r == T()) return complex<T>();
  const T a = z.real();
  const T b = z.imag();
  const T re = ::ft_shim::sqrt((r + a) / T(2));
  const T im = (b < T() ? T(-1) : T(1)) * ::ft_shim::sqrt((r - a) / T(2));
  return complex<T>(re, im);
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> pow(
    const complex<T>& base, const complex<T>& exponent) {
  return exp(exponent * log(base));
}
template <typename T, typename U>
FT_THRUST_SHIM_HOST_DEVICE complex<T> pow(
    const complex<T>& base, const U& exponent) {
  return pow(base, complex<T>(static_cast<T>(exponent)));
}
template <typename T, typename U>
FT_THRUST_SHIM_HOST_DEVICE complex<T> pow(
    const U& base, const complex<T>& exponent) {
  return ::ft_shim::pow(complex<T>(static_cast<T>(base)), exponent);
}

namespace detail {

template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> _sin_impl(const complex<T>& z) {
  // ::ft_shim::sin(a+bi) = ::ft_shim::sin(a)::ft_shim::cosh(b) + i ::ft_shim::cos(a)::ft_shim::sinh(b)
  return complex<T>(
      ::ft_shim::sin(z.real()) * ::ft_shim::cosh(z.imag()), ::ft_shim::cos(z.real()) * ::ft_shim::sinh(z.imag()));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> _cos_impl(const complex<T>& z) {
  // ::ft_shim::cos(a+bi) = ::ft_shim::cos(a)::ft_shim::cosh(b) - i ::ft_shim::sin(a)::ft_shim::sinh(b)
  return complex<T>(
      ::ft_shim::cos(z.real()) * ::ft_shim::cosh(z.imag()), -::ft_shim::sin(z.real()) * ::ft_shim::sinh(z.imag()));
}

}  // namespace detail

template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> sin(const complex<T>& z) {
  return detail::_sin_impl(z);
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> cos(const complex<T>& z) {
  return detail::_cos_impl(z);
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> tan(const complex<T>& z) {
  return detail::_sin_impl(z) / detail::_cos_impl(z);
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> asin(const complex<T>& z) {
  const complex<T> i(T(), T(1));
  return -(i * log(i * z + sqrt(complex<T>(T(1)) - z * z)));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> acos(const complex<T>& z) {
  const complex<T> i(T(), T(1));
  return -(i * log(z + i * sqrt(complex<T>(T(1)) - z * z)));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> atan(const complex<T>& z) {
  const complex<T> i(T(), T(1));
  const complex<T> one(T(1));
  return (i / T(2)) * (log(one - i * z) - log(one + i * z));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> sinh(const complex<T>& z) {
  // ::ft_shim::sinh(a+bi) = ::ft_shim::sinh(a)::ft_shim::cos(b) + i ::ft_shim::cosh(a)::ft_shim::sin(b)
  return complex<T>(
      ::ft_shim::sinh(z.real()) * ::ft_shim::cos(z.imag()), ::ft_shim::cosh(z.real()) * ::ft_shim::sin(z.imag()));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> cosh(const complex<T>& z) {
  // ::ft_shim::cosh(a+bi) = ::ft_shim::cosh(a)::ft_shim::cos(b) + i ::ft_shim::sinh(a)::ft_shim::sin(b)
  return complex<T>(
      ::ft_shim::cosh(z.real()) * ::ft_shim::cos(z.imag()), ::ft_shim::sinh(z.real()) * ::ft_shim::sin(z.imag()));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> tanh(const complex<T>& z) {
  return sinh(z) / cosh(z);
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> asinh(const complex<T>& z) {
  return log(z + sqrt(complex<T>(T(1)) + z * z));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> acosh(const complex<T>& z) {
  return log(z + sqrt(z * z - complex<T>(T(1))));
}
template <typename T>
FT_THRUST_SHIM_HOST_DEVICE complex<T> atanh(const complex<T>& z) {
  const complex<T> one(T(1));
  return (log(one + z) - log(one - z)) / T(2);
}

}  // namespace thrust
