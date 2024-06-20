// RUN: safe-c %s -- | FileCheck %s --match-full-lines

#define unsafe if (0) ; else

int main(void) {
        unsafe {
                1;
        }
        return 0;
}

// CHECK:   hl.translation_unit {
// CHECK:     hl.typedef "__int128_t" : !hl.int128
// CHECK:     hl.typedef "__uint128_t" : !hl.int128< unsigned >
// CHECK:     hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
// CHECK:     hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
// CHECK:     hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
// CHECK:     hl.func @main external () -> !hl.int {
// CHECK:       core.scope {
// CHECK:         safety.unsafe {
// CHECK:           core.scope {
// CHECK:             %1 = hl.const #core.integer<1> : !hl.int
// CHECK:           }
// CHECK:         }
// CHECK:         core.scope {
// CHECK:           %1 = hl.const #core.integer<1> : !hl.int
// CHECK:         }
// CHECK:         %0 = hl.const #core.integer<0> : !hl.int
// CHECK:         hl.return %0 : !hl.int
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }
