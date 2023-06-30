// RUN: macronify %s | FileCheck %s --match-full-lines

//CHECK: hl.translation_unit {
//CHECK:   hl.typedef "__int128_t" : !hl.int128
//CHECK:   hl.typedef "__uint128_t" : !hl.int128< unsigned >
//CHECK:   hl.struct "__NSConstantString_tag" : {
//CHECK:     hl.field "isa" : !hl.ptr<!hl.int< const >>
//CHECK:     hl.field "flags" : !hl.int
//CHECK:     hl.field "str" : !hl.ptr<!hl.char< const >>
//CHECK:     hl.field "length" : !hl.long
//CHECK:   }
//CHECK:   hl.typedef "__NSConstantString" : !hl.record<"__NSConstantString_tag">
//CHECK:   hl.typedef "__builtin_ms_va_list" : !hl.ptr<!hl.char>
//CHECK:   hl.struct "__va_list_tag" : {
//CHECK:     hl.field "gp_offset" : !hl.int< unsigned >
//CHECK:     hl.field "fp_offset" : !hl.int< unsigned >
//CHECK:     hl.field "overflow_arg_area" : !hl.ptr<!hl.void>
//CHECK:     hl.field "reg_save_area" : !hl.ptr<!hl.void>
//CHECK:   }
//CHECK:   hl.typedef "__builtin_va_list" : !hl.array<1, !hl.record<"__va_list_tag">>
//CHECK:   hl.typedef "size_t" : !hl.long< unsigned >
//CHECK:   hl.struct "list_head" : {
//CHECK:     hl.field "next" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:     hl.field "prev" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:   }
//CHECK:   hl.func internal @list_is_head (%arg0: !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>, %arg1: !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>) -> !hl.int attributes {sym_visibility = "private"} {
//CHECK:     hl.scope {
//CHECK:       %0 = hl.ref %arg0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>
//CHECK:       %1 = hl.implicit_cast %0 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>
//CHECK:       %2 = hl.ref %arg1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>>
//CHECK:       %3 = hl.implicit_cast %2 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>
//CHECK:       %4 = hl.cmp eq %1, %3 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >>, !hl.ptr<!hl.elaborated<!hl.record<"list_head">,  const >> -> !hl.bool
//CHECK:       %5 = hl.implicit_cast %4 IntegralCast : !hl.bool -> !hl.int
//CHECK:       hl.return %5 : !hl.int
//CHECK:     }
//CHECK:   }
//CHECK:   hl.func external @main () -> !hl.int {
//CHECK:     hl.scope {
//CHECK:       %0 = hl.var "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:       %1 = hl.var "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:       hl.scope {
//CHECK:         %3 = macroni.parameter "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
//CHECK:           %9 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:           hl.value.yield %9 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:         }
//CHECK:         %4 = hl.expr : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
//CHECK:           %9 = macroni.parameter "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
//CHECK:             %10 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:             hl.value.yield %10 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:           }
//CHECK:           hl.value.yield %9 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:         }
//CHECK:         %5 = hl.implicit_cast %4 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:         %6 = hl.member %5 at "next" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:         %7 = hl.implicit_cast %6 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:         %8 = hl.assign %7 to %3 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>, !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:         hl.for {
//CHECK:           %9 = macroni.parameter "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
//CHECK:             %18 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:             hl.value.yield %18 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:           }
//CHECK:           %10 = hl.implicit_cast %9 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:           %11 = hl.implicit_cast %10 NoOp : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:           %12 = hl.expr : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
//CHECK:             %18 = macroni.parameter "head" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
//CHECK:               %19 = hl.ref %0 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:               hl.value.yield %19 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:             }
//CHECK:             hl.value.yield %18 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:           }
//CHECK:           %13 = hl.implicit_cast %12 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:           %14 = hl.implicit_cast %13 NoOp : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:           %15 = hl.call @list_is_head(%11, %14) : (!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>, !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>) -> !hl.int
//CHECK:           %16 = hl.implicit_cast %15 IntegralToBoolean : !hl.int -> !hl.bool
//CHECK:           %17 = hl.lnot %16 : !hl.bool -> !hl.bool
//CHECK:           hl.cond.yield %17 : !hl.bool
//CHECK:         } incr {
//CHECK:           %9 = macroni.parameter "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
//CHECK:             %15 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:             hl.value.yield %15 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:           }
//CHECK:           %10 = macroni.parameter "pos" : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> {
//CHECK:             %15 = hl.ref %1 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:             hl.value.yield %15 : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:           }
//CHECK:           %11 = hl.implicit_cast %10 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:           %12 = hl.member %11 at "next" : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>> -> !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>>
//CHECK:           %13 = hl.implicit_cast %12 LValueToRValue : !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:           %14 = hl.assign %13 to %9 : !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>, !hl.lvalue<!hl.ptr<!hl.elaborated<!hl.record<"list_head">>>> -> !hl.ptr<!hl.elaborated<!hl.record<"list_head">>>
//CHECK:         } do {
//CHECK:           hl.skip
//CHECK:         }
//CHECK:       }
//CHECK:       %2 = hl.const #hl.integer<0> : !hl.int
//CHECK:       hl.return %2 : !hl.int
//CHECK:     }
//CHECK:   }
//CHECK: }

typedef unsigned long size_t;

struct list_head {
        struct list_head *next, *prev;
};

static inline int list_is_head(const struct list_head *list, const struct list_head *head) {
        return list == head;
}

#define list_for_each(pos, head) \
	for (pos = (head)->next; !list_is_head(pos, (head)); pos = pos->next)

#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

#define container_of(ptr, type, member) ({          \
	const typeof(((type *)0)->member)*__mptr = (ptr);    \
		     (type *)((char *)__mptr - offsetof(type, member)); })

#define list_entry(ptr, type, member) \
	container_of(ptr, type, member)

#define list_first_entry(ptr, type, member) \
	list_entry((ptr)->next, type, member)

#define list_for_each_entry(pos, head, member)				\
	for (pos = list_first_entry(head, typeof(*pos), member);	\
	     !list_entry_is_head(pos, head, member);			\
	     pos = list_next_entry(pos, member))

int main(void) {
        struct list_head *head, *pos;
        list_for_each(pos, head);
        return 0;
}
