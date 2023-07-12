// Before
hl.func external @rcu_read_lock () {
  hl.scope {
    hl.return
  }
}
hl.func external @rcu_read_unlock () {
  hl.scope {
    hl.return
  }
}
hl.func external @foo () {
  hl.scope {
    hl.return
  }
}
hl.func external @main () -> !hl.int {
  hl.scope {
    %0 = hl.var "i" : !hl.lvalue<!hl.int> = {
      %2 = hl.const #hl.integer<0> : !hl.int
      hl.value.yield %2 : !hl.int
    }
    hl.call @rcu_read_lock() : () -> ()
    hl.scope {
      %2 = hl.ref %0 : !hl.lvalue<!hl.int>
      %3 = hl.const #hl.integer<0> : !hl.int
      %4 = hl.assign %3 to %2 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
      hl.for {
        %5 = hl.ref %0 : !hl.lvalue<!hl.int>
        %6 = hl.implicit_cast %5 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
        %7 = hl.const #hl.integer<10> : !hl.int
        %8 = hl.cmp slt %6, %7 : !hl.int, !hl.int -> !hl.bool
        hl.cond.yield %8 : !hl.bool
      } incr {
        %5 = hl.ref %0 : !hl.lvalue<!hl.int>
        %6 = hl.post.inc %5 : !hl.lvalue<!hl.int> -> !hl.int
      } do {
        hl.scope {
          hl.call @foo() : () -> ()
        }
      }
    }
    hl.call @rcu_read_unlock() : () -> ()
    %1 = hl.const #hl.integer<0> : !hl.int
    hl.return %1 : !hl.int
  }
}

// After
hl.func external @rcu_read_lock () {
  hl.scope {
    hl.return
  }
}
hl.func external @rcu_read_unlock () {
  hl.scope {
    hl.return
  }
}
hl.func external @foo () {
  hl.scope {
    hl.return
  }
}
hl.func external @main () -> !hl.int {
  hl.scope {
    %0 = hl.var "i" : !hl.lvalue<!hl.int> = {
      %2 = hl.const #hl.integer<0> : !hl.int
      hl.value.yield %2 : !hl.int
    }
    kernel.rcu.critical_section {
      hl.scope {
        %2 = hl.ref %0 : !hl.lvalue<!hl.int>
        %3 = hl.const #hl.integer<0> : !hl.int
        %4 = hl.assign %3 to %2 : !hl.int, !hl.lvalue<!hl.int> -> !hl.int
        hl.for {
          %5 = hl.ref %0 : !hl.lvalue<!hl.int>
          %6 = hl.implicit_cast %5 LValueToRValue : !hl.lvalue<!hl.int> -> !hl.int
          %7 = hl.const #hl.integer<10> : !hl.int
          %8 = hl.cmp slt %6, %7 : !hl.int, !hl.int -> !hl.bool
          hl.cond.yield %8 : !hl.bool
        } incr {
          %5 = hl.ref %0 : !hl.lvalue<!hl.int>
          %6 = hl.post.inc %5 : !hl.lvalue<!hl.int> -> !hl.int
        } do {
          hl.scope {
            hl.call @foo() : () -> ()
          }
        }
      }
    }
    %1 = hl.const #hl.integer<0> : !hl.int
    hl.return %1 : !hl.int
  }
}
