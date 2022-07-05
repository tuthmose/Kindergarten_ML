def main_ga_loop(ngeneration, population, sel_press, pCO, pMut, sspace, ffunc, **kwargs):
    """
    Applies the GA loop to the input population and returns the best
    chromosome, its fitness and the final population
    """
    if not "tsize" in kwargs.keys():
        kwargs["tsize"] = 2
    if not "vstep" in kwargs.keys():
        kwargs["vstep"] = 50
    if not "sel_meth" in kwargs.keys():
        kwargs["sel_meth"] = "rank"
    if not "co_meth" in kwargs.keys():
        kwargs["co_meth"] = "interp"
    if not "alpha" in kwargs.keys():
        kwargs["alpha"] = 0.5        
    if not "mut_meth" in kwargs.keys():
        kwargs["mut_meth"] = "const"
    if not "tol" in kwargs.keys():
        kwargs["tol"] = 1e-5  
    if not "ffkwds" in kwargs.keys():
        kwargs["ffkwds"] = dict()          
    fitness, population = calc_fitness(min_best, ffunc, population, kwargs["ffkwds"])
    best_previous = fitness[0]
    best = list()
    best_f = list()
    for gen in range(ngeneration):
        # select fitter specimens for reproduction
        nmating, parents = ParentSelection(sel_press, kwargs["sel_meth"], population, kwargs["tsize"])
        # generate offspring and check cross over probability
        offspring = crossover(pCO, kwargs["alpha"], nmating, parents, population, method=kwargs["co_meth"])
        population = np.concatenate((population, offspring))   
        # mutate
        mutated, population = mutation(pMut, kwargs["mut_meth"], population, sspace)
        # calculate fitness
        fitness, population = calc_fitness(min_best, ffunc, population, kwargs["ffkwds"])
        if gen % kwargs["vstep"]== 0:
            print("Generation ",gen, " best specimen ",fitness[0])        
        # remove less fit individuals
        population = fit_selection(nmating, population)
        gain = abs(fitness[0]-best_previous)
        if kwargs['tol'] > 0:
            if gen > kwargs["vstep"] and gain < kwargs["tol"]:
                print("Gain/loss less than tolerance: ",gain)
                break
        best.append(population[0])
        best_f.append(fitness[0])
    return gen, best_f, best