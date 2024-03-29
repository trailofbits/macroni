// RUN: kernelize -x c %s | FileCheck %s --match-full-lines

int barrier(void) { return 0; }
#define smp_mb() barrier()

int main(void) {
        smp_mb();
        return 0;
}

// CHECK:hl.translation_unit {
// CHECK:  hl.typedef "__int128_t" : !hl.int128
// CHECK:  hl.typedef "__uint128_t" : !hl.int128< unsigned >
// CHECK:  hl.struct "__NSConstantString_tag" {type_visibility = #unsup<attr "type_visibility">} : {
// CHECK:    hl.field "isa" : !hl.ptr<!hl.int< const >>
// CHECK:    hl.field "flags" : !hl.int
// CHECK:    hl.field "str" : !hl.ptr<!hl.char< const >>
// CHECK:    hl.field "length" : !hl.long
// CHECK:  }
// CHECK:  hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
// CHECK:  hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
// CHECK:  hl.struct "__va_list_tag" {type_visibility = #unsup<attr "type_visibility">} : {
// CHECK:    hl.field "gp_offset" : !hl.int< unsigned >
// CHECK:    hl.field "fp_offset" : !hl.int< unsigned >
// CHECK:    hl.field "overflow_arg_area" : !hl.ptr<!hl.void>
// CHECK:    hl.field "reg_save_area" : !hl.ptr<!hl.void>
// CHECK:  }
// CHECK:  hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
// CHECK:  hl.func @barrier () -> !hl.int {
// CHECK:    core.scope {
// CHECK:      %0 = hl.const #core.integer<0> : !hl.int
// CHECK:      hl.return %0 : !hl.int
// CHECK:    }
// CHECK:  }
// CHECK:  hl.func @main () -> !hl.int {
// CHECK:    core.scope {
// CHECK:      %0 = kernel.smp_mb() : () -> !hl.int
// CHECK:      %1 = hl.const #core.integer<0> : !hl.int
// CHECK:      hl.return %1 : !hl.int
// CHECK:    }
// CHECK:  }
// CHECK:}

