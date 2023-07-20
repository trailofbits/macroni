// RUN: macronify -xc %s --convert | FileCheck %s --match-full-lines

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
// CHECK:   hl.func external @main () -> !hl.int {
// CHECK:     %0 = hl.var "x" : !hl.lvalue<!hl.int>
// CHECK:     %1 = hl.var "ptr" : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:     %2 = macroni.parameter "x" : !hl.lvalue<!hl.int> {
// CHECK:       %6 = hl.ref %0 : !hl.lvalue<!hl.int>
// CHECK:       hl.value.yield %6 : !hl.lvalue<!hl.int>
// CHECK:     }
// CHECK:     %3 = macroni.parameter "ptr" : !hl.lvalue<!hl.ptr<!hl.int>> {
// CHECK:       %6 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:       hl.value.yield %6 : !hl.lvalue<!hl.ptr<!hl.int>>
// CHECK:     }
// CHECK:     %4 = kernel.get_user get_user(%2, %3) : (!hl.lvalue<!hl.int>, !hl.lvalue<!hl.ptr<!hl.int>>) -> !hl.int
// CHECK:     %5 = hl.const #hl.integer<0> : !hl.int
// CHECK:     hl.return %5 : !hl.int
// CHECK:   }
// CHECK: }

#define get_user(x, ptr) ({ int res = 0; if (1) { x = *ptr; res = 1; } res; })

int main(void) {
        int x, *ptr;
        get_user(x, ptr);
        return 0;
}