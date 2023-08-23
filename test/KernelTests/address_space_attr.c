// RUN: kernelize -x c %s | FileCheck %s --match-full-lines

// CHECK: hl.translation_unit {
// CHECK:   hl.typedef "__int128_t" : !hl.int128
// CHECK:   hl.typedef "__uint128_t" : !hl.int128< unsigned >
// CHECK:   hl.struct "__NSConstantString_tag" : {
// CHECK:     hl.field "isa" : !hl.ptr<!hl.int< const >>
// CHECK:     hl.field "flags" : !hl.int
// CHECK:     hl.field "str" : !hl.ptr<!hl.char< const >>
// CHECK:     hl.field "length" : !hl.long
// CHECK:   }
// CHECK:   hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
// CHECK:   hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
// CHECK:   hl.struct "__va_list_tag" : {
// CHECK:     hl.field "gp_offset" : !hl.int< unsigned >
// CHECK:     hl.field "fp_offset" : !hl.int< unsigned >
// CHECK:     hl.field "overflow_arg_area" : !hl.ptr<!hl.void>
// CHECK:     hl.field "reg_save_area" : !hl.ptr<!hl.void>
// CHECK:   }
// CHECK:   hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
// CHECK:   %0 = hl.var "a" : !hl.lvalue<!kernel<address_space(0) <!hl.attributed<!hl.int>>>>
// CHECK:   %1 = hl.var "b" : !hl.lvalue<!kernel<address_space(1) <!hl.attributed<!hl.int>>>>
// CHECK:   %2 = hl.var "c" : !hl.lvalue<!kernel<address_space(2) <!hl.attributed<!hl.int>>>>
// CHECK:   %3 = hl.var "d" : !hl.lvalue<!kernel<address_space(3) <!hl.attributed<!hl.int>>>>
// CHECK:   %4 = hl.var "e" : !hl.lvalue<!kernel<address_space(4) <!hl.attributed<!hl.int>>>>
// CHECK: }

#define addr_space(x)   __attribute__((address_space(x)))
int a addr_space(0);
int b addr_space(1);
int c addr_space(2);
int d addr_space(3);
int e addr_space(4);
