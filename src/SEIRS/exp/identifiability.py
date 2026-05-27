from juliacall import Main as jl

jl.seval("using StructuralIdentifiability")

results = jl.seval("""
    ODE = @ODEmodel(

        S'(t) =
            beta(t)*I(t)*S(t)/(S(t)+E(t)+I(t)+R(t))
            - 0.0003671*S(t)
            + 0.0006762*(S(t)+E(t)+I(t)+R(t))
            + 0.0027400*R(t),

        E'(t) =
            beta(t)*I(t)*S(t)/(S(t)+E(t)+I(t)+R(t))
            - (0.0003671 + sigma(t) + 0.3500000)*E(t),

        I'(t) =
            sigma(t)*E(t)
            - (0.0003671 + 0.0300000 + 0.0001500)*I(t),

        R'(t) =
            0.3500000*E(t)
            + 0.0001500*I(t)
            - 0.0003671*R(t)
            - 0.0027400*R(t),

        beta'(t) = u1(t),

        sigma'(t) = u2(t),

        y(t) = I(t)
    )

    assess_local_identifiability(ODE)
""")

print(results)
#OrderedCollections.OrderedDict{Any, Bool}(S(t) => true, E(t) => true, I(t) => true, R(t) => true, beta(t) => true, sigma(t) => true)