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
// CHECK:   hl.typedef "size_t" : !hl.long< unsigned >
// CHECK:   hl.struct "location" : {
// CHECK:     hl.field "file" : !hl.ptr<!hl.char>
// CHECK:     hl.field "line" : !hl.int
// CHECK:     hl.field "col" : !hl.int
// CHECK:   }
// CHECK:   hl.func external @main () -> !hl.int {
// CHECK:     hl.scope {
// CHECK:       %0 = kernel.offsetof offsetof(!hl.elaborated<!hl.record<"location">>, "line") : () -> !hl.elaborated<!hl.typedef<"size_t">>
// CHECK:       %1 = hl.const #hl.integer<0> : !hl.int
// CHECK:       hl.return %1 : !hl.int
// CHECK:     }
// CHECK:   }
// CHECK: }

typedef unsigned long int size_t;

#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

struct location {
        char *file;
        int line;
        int col;
};

int main(void) {
        offsetof(struct location, line);
        return 0;
}
