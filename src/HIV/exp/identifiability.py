from juliacall import Main as jl

jl.seval("using StructuralIdentifiability")

results = jl.seval("""
    ODE = @ODEmodel(

        TU'(t) = lambda
                  - rho*TU(t)
                  - eta(t)*TU(t)*V(t),

        TI'(t) = eta(t)*TU(t)*V(t)
                  - delta*TI(t),

        V'(t)  = N*delta*TI(t)
                  - c*V(t),

        eta'(t) = u(t),

        y1(t) = TU(t) + TI(t),

        y2(t) = V(t)
    )

    assess_local_identifiability(ODE)
""")

print(results)
#output: OrderedCollections.OrderedDict{Any, Bool}(TU(t) => true, TI(t) => true, V(t) => true, eta(t) => true, N => true, c => true, delta => true, lambda => true, rho => true)