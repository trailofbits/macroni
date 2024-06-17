#ifndef RCU
#define RCU

typedef unsigned long uintptr_t;

static inline int rcu_read_lock_held(void) { return 1; }

#undef NULL
#define NULL ((void *)0)

#define ___PASTE(a, b) a##b
#define __PASTE(a, b) ___PASTE(a, b)

#define __UNIQUE_ID(prefix) __PASTE(__PASTE(__UNIQUE_ID_, prefix), __COUNTER__)

#ifdef __CHECKER__
#define rcu_check_sparse(p, space) ((void)(((typeof(*p) space *)p) == p))
#else /* #ifdef __CHECKER__ */
#define rcu_check_sparse(p, space)
#endif /* #else #ifdef __CHECKER__ */

// NOTE(Brent): smp_store_release is defined multiple times throughout the
// kernel, so we can't reliably match on its expanded AST. We can match on the
// expansion itself though.
#ifndef smp_store_release
#define smp_store_release(p, v)                                                \
  do {                                                                         \
    barrier();                                                                 \
    WRITE_ONCE(*p, v);                                                         \
  } while (0)
#endif

#ifdef CONFIG_PROVE_RCU

#define RCU_LOCKDEP_WARN(c, s)                                                 \
  do {                                                                         \
    static bool __section(".data.unlikely") __warned;                          \
    if (debug_lockdep_rcu_enabled() && (c) && debug_lockdep_rcu_enabled() &&   \
        !__warned) {                                                           \
      __warned = true;                                                         \
      lockdep_rcu_suspicious(__FILE__, __LINE__, s);                           \
    }                                                                          \
  } while (0)

#if defined(CONFIG_PROVE_RCU) && !defined(CONFIG_PREEMPT_RCU)
static inline void rcu_preempt_sleep_check(void) {
  RCU_LOCKDEP_WARN(lock_is_held(&rcu_lock_map),
                   "Illegal context switch in RCU read-side critical section");
}
#else  /* #ifdef CONFIG_PROVE_RCU */
static inline void rcu_preempt_sleep_check(void) {}
#endif /* #else #ifdef CONFIG_PROVE_RCU */

#define rcu_sleep_check()                                                      \
  do {                                                                         \
    rcu_preempt_sleep_check();                                                 \
    if (!IS_ENABLED(CONFIG_PREEMPT_RT))                                        \
      RCU_LOCKDEP_WARN(                                                        \
          lock_is_held(&rcu_bh_lock_map),                                      \
          "Illegal context switch in RCU-bh read-side critical section");      \
    RCU_LOCKDEP_WARN(                                                          \
        lock_is_held(&rcu_sched_lock_map),                                     \
        "Illegal context switch in RCU-sched read-side critical section");     \
  } while (0)

#else /* #ifdef CONFIG_PROVE_RCU */

#define RCU_LOCKDEP_WARN(c, s)                                                 \
  do {                                                                         \
  } while (0 && (c))
#define rcu_sleep_check()                                                      \
  do {                                                                         \
  } while (0)

#endif /* #else #ifdef CONFIG_PROVE_RCU */

#define __scalar_type_to_expr_cases(type)                                      \
  unsigned type : (unsigned type)0, signed type : (signed type)0

#define __unqual_scalar_typeof(x)                                              \
  typeof(_Generic((x),                                                         \
      char: (char)0,                                                           \
      __scalar_type_to_expr_cases(char),                                       \
      __scalar_type_to_expr_cases(short),                                      \
      __scalar_type_to_expr_cases(int),                                        \
      __scalar_type_to_expr_cases(long),                                       \
      __scalar_type_to_expr_cases(long long),                                  \
      default: (x)))

#ifndef __READ_ONCE
#define __READ_ONCE(x) (*(const volatile __unqual_scalar_typeof(x) *)&(x))
#endif

typedef __signed__ char __s8;
typedef unsigned char __u8;

typedef __signed__ short __s16;
typedef unsigned short __u16;

typedef __signed__ int __s32;
typedef unsigned int __u32;

#ifdef __GNUC__
__extension__ typedef __signed__ long long __s64;
__extension__ typedef unsigned long long __u64;
#else
typedef __signed__ long long __s64;
typedef unsigned long long __u64;
#endif

#ifndef __always_inline
#define __always_inline inline __attribute__((always_inline))
#endif

/* Optimization barrier */
/* The "volatile" is due to gcc bugs */
#define barrier() __asm__ __volatile__("" : : : "memory")

typedef __u8 __attribute__((__may_alias__)) __u8_alias_t;
typedef __u16 __attribute__((__may_alias__)) __u16_alias_t;
typedef __u32 __attribute__((__may_alias__)) __u32_alias_t;
typedef __u64 __attribute__((__may_alias__)) __u64_alias_t;

static __always_inline void __write_once_size(volatile void *p, void *res,
                                              int size) {
  switch (size) {
  case 1:
    *(volatile __u8_alias_t *)p = *(__u8_alias_t *)res;
    break;
  case 2:
    *(volatile __u16_alias_t *)p = *(__u16_alias_t *)res;
    break;
  case 4:
    *(volatile __u32_alias_t *)p = *(__u32_alias_t *)res;
    break;
  case 8:
    *(volatile __u64_alias_t *)p = *(__u64_alias_t *)res;
    break;
  default:
    barrier();
    __builtin_memcpy((void *)p, (const void *)res, size);
    barrier();
  }
}

#define WRITE_ONCE(x, val)                                                     \
  ({                                                                           \
    union {                                                                    \
      typeof(x) __val;                                                         \
      char __c[1];                                                             \
    } __u = {.__val = (val)};                                                  \
    __write_once_size(&(x), __u.__c, sizeof(x));                               \
    __u.__val;                                                                 \
  })

#define __native_word(t)                                                       \
  (sizeof(t) == sizeof(char) || sizeof(t) == sizeof(short) ||                  \
   sizeof(t) == sizeof(int) || sizeof(t) == sizeof(long))

#if __has_attribute(__error__)
#define __compiletime_error(msg) __attribute__((__error__(msg)))
#else
#define __compiletime_error(msg)
#endif

#define __noreturn __attribute__((__noreturn__))

#ifdef __OPTIMIZE__
#define __compiletime_assert(condition, msg, prefix, suffix)                   \
  do {                                                                         \
    /*                                                                         \
     * __noreturn is needed to give the compiler enough                        \
     * information to avoid certain possibly-uninitialized                     \
     * warnings (regardless of the build failing).                             \
     */                                                                        \
    __noreturn extern void prefix##suffix(void) __compiletime_error(msg);      \
    if (!(condition))                                                          \
      prefix##suffix();                                                        \
  } while (0)
#else
#define __compiletime_assert(condition, msg, prefix, suffix)                   \
  do {                                                                         \
  } while (0)
#endif

#define _compiletime_assert(condition, msg, prefix, suffix)                    \
  __compiletime_assert(condition, msg, prefix, suffix)

#define compiletime_assert(condition, msg)                                     \
  _compiletime_assert(condition, msg, __compiletime_assert_, __COUNTER__)

#define compiletime_assert_rwonce_type(t)                                      \
  compiletime_assert(__native_word(t) || sizeof(t) == sizeof(long long),       \
                     "Unsupported access size for {READ,WRITE}_ONCE().")

#define READ_ONCE(x)                                                           \
  ({                                                                           \
    compiletime_assert_rwonce_type(x);                                         \
    __READ_ONCE(x);                                                            \
  })

#if defined(CONFIG_DEBUG_INFO_BTF) && defined(CONFIG_PAHOLE_HAS_BTF_TAG) &&    \
    __has_attribute(btf_type_tag) && !defined(__BINDGEN__)
#define BTF_TYPE_TAG(value) __attribute__((btf_type_tag(#value)))
#else
#define BTF_TYPE_TAG(value) /* nothing */
#endif

#ifdef __CHECKER__
#define __kernel __attribute__((address_space(0)))
#define __rcu __attribute__((noderef, address_space(__rcu)))
#define __force __attribute__((force))
#else /* __CHECKER__ */
#define __kernel
#define __rcu BTF_TYPE_TAG(rcu)
#define __force
#endif /* __CHECKER__ */

#define RCU_INITIALIZER(v) (typeof(*(v)) __force __rcu *)(v)

#define __rcu_dereference_check(p, local, c, space)                            \
  ({                                                                           \
    /* Dependency order vs. p above. */                                        \
    typeof(*p) *local = (typeof(*p) *__force)READ_ONCE(p);                     \
    RCU_LOCKDEP_WARN(!(c), "suspicious rcu_dereference_check() usage");        \
    rcu_check_sparse(p, space);                                                \
    ((typeof(*p) __force __kernel *)(local));                                  \
  })

#define rcu_dereference_check(p, c)                                            \
  __rcu_dereference_check((p), __UNIQUE_ID(rcu), (c) || rcu_read_lock_held(),  \
                          __rcu)

#define rcu_dereference(p) rcu_dereference_check(p, 0)

#define rcu_assign_pointer(p, v)                                               \
  do {                                                                         \
    uintptr_t _r_a_p__v = (uintptr_t)(v);                                      \
    rcu_check_sparse(p, __rcu);                                                \
                                                                               \
    if (__builtin_constant_p(v) && (_r_a_p__v) == (uintptr_t)NULL)             \
      WRITE_ONCE((p), (typeof(p))(_r_a_p__v));                                 \
    else                                                                       \
      smp_store_release(&p, RCU_INITIALIZER((typeof(p))_r_a_p__v));            \
  } while (0)

// NOTE(Brent): rcu_assign_pointer() is also defined like this in one place in
// the kernel:
//
// #define rcu_assign_pointer(p, v) do { (p) = (v); } while (0)
//
// However, this not the definition that kernel code will likely be using, so we
// don't have to match on it. We could add an alternative matcher just to be
// safe though.

#define __rcu_access_pointer(p, local, space)                                  \
  ({                                                                           \
    typeof(*p) *local = (typeof(*p) *__force)READ_ONCE(p);                     \
    rcu_check_sparse(p, space);                                                \
    ((typeof(*p) __force __kernel *)(local));                                  \
  })

#define rcu_access_pointer(p) __rcu_access_pointer((p), __UNIQUE_ID(rcu), __rcu)

#define __rcu_dereference_protected(p, local, c, space)                        \
  ({                                                                           \
    RCU_LOCKDEP_WARN(!(c), "suspicious rcu_dereference_protected() usage");    \
    rcu_check_sparse(p, space);                                                \
    ((typeof(*p) __force __kernel *)(p));                                      \
  })

#define rcu_dereference_protected(p, c)                                        \
  __rcu_dereference_protected((p), __UNIQUE_ID(rcu), (c), __rcu)

#define rcu_replace_pointer(rcu_ptr, ptr, c)                                   \
  ({                                                                           \
    typeof(ptr) __tmp = rcu_dereference_protected((rcu_ptr), (c));             \
    rcu_assign_pointer((rcu_ptr), (ptr));                                      \
    __tmp;                                                                     \
  })

#endif
