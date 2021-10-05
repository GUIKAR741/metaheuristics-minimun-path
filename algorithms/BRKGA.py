"""Genetic Algorithm."""
import random

import numpy

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

from contextlib import contextmanager
from datetime import datetime, timedelta

from math import ceil, fabs


@contextmanager
def timeit(file_write=None):
    """Context Manager to check runtime."""
    start_time = datetime.now()
    print(f'Tempo de Inicio (hh:mm:ss.ms) {start_time}', file=file_write)
    yield
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    print(f'Tempo de Termino (hh:mm:ss.ms) {end_time}', file=file_write)
    print(f'Tempo Total (hh:mm:ss.ms) {time_elapsed}', file=file_write)


def dist2pt(x1, y1, x2, y2):
    """."""
    return max(fabs(x2 - x1), fabs(y2 - y1))  # Chebyschev Distance


def midPoint(x1, y1, x2, y2):
    """."""
    return (x1 + x2) / 2, (y1 + y2) / 2


def plotar(indiv, f):
    """."""
    individuo = decode(indiv)
    fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    x1, y1, x, y = [], [], [], []
    colors = ['red', 'gray']
    cutA = 1
    i1 = individuo[0][0]
    a1 = edges[i1] if individuo[1][0] == 0 else edges[i1][::-1]
    deslocamentos = []
    x.append(a1[0][0])
    y.append(a1[0][1])
    x.append(a1[1][0])
    y.append(a1[1][1])
    f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0], scale_units='xy', angles='xy', scale=1, color=colors[0])
    f1_axes.annotate(str(cutA), midPoint(*a1[0], *a1[1]))
    cutA += 1
    for i in range(len(individuo[0]) - 1):
        i1 = individuo[0][i]  # current edge
        i2 = individuo[0][i + 1 if i + 1 < len(individuo[0]) else 0]  # next edge
        a1 = edges[i1] if individuo[1][i] == 0 else edges[i1][::-1]  # current edge
        a2 = edges[i2] if individuo[1][
            i + 1 if i + 1 < len(individuo[0]) else 0] == 0 else edges[i2][::-1]  # next edge
        x1, y1, x, y = [], [], [], []
        if a1[1] != a2[0]:  # if the next edge not starts when the first end
            x1.append(a1[1][0])
            y1.append(a1[1][1])
            x1.append(a2[0][0])
            y1.append(a2[0][1])
            deslocamentos.append({
                'pontos': [x1[0], y1[0], x1[1] - x1[0], y1[1] - y1[0]],
                'annot': str(cutA),
                'mid': midPoint(*a1[1], *a2[0])
            })
            cutA += 1
        # plot next edge
        x.append(a2[0][0])
        y.append(a2[0][1])
        x.append(a2[1][0])
        y.append(a2[1][1])
        f1_axes.annotate(str(cutA), midPoint(*a2[0], *a2[1]))
        f1_axes.quiver(x[0], y[0], x[1] - x[0], y[1] - y[0], scale_units='xy', angles='xy', scale=1, color=colors[0])
        cutA += 1
    for i in deslocamentos:
        f1_axes.annotate(i['annot'], (i['mid'][0] - 3, i['mid'][1]))
        f1_axes.quiver(*i['pontos'], width=.005, scale_units='xy', angles='xy', scale=1, color=colors[1])
    f1_axes.set_xlim(*f1_axes.get_xlim())
    f1_axes.set_ylim(*f1_axes.get_ylim())
    plt.title("Tempo Requerido: {:.2f}".format(indiv.fitness.values[0]))
    fig1.savefig(f'plots/brkga/{f}.png', dpi=300)
    plt.close()


def genIndividuo(edges):
    """
    Generate Individuo.

    args:
        edges -> edges to cut of grapth

    individuo[0]: order of edges
    individuo[1]: order of cut

    """
    v = [random.randint(0, 1) for i in range(len(edges))]
    random.shuffle(v)
    return random.sample(range(len(edges)), len(edges)), v


def genIndividuoRK(edges):
    """
    Generate Individuo.

    args:
        edges -> edges to cut of grapth

    individuo[0]: order of edges
    individuo[1]: order of cut

    """
    return [random.random() for i in range(len(edges))], [
        random.random() for i in range(len(edges))]


def decode(ind):
    """."""
    return [ind[0].index(i) for i in sorted(ind[0])], [0 if i < 0.5 else 1 for i in ind[1]]


def evalCut(individuo, pi=100 / 6, mi=400):
    """
    Eval Edges Cut.

    args:
        pi -> cutting speed
        mi -> travel speed

    if individuo[1][i] == 0 the cut is in edge order
    else the cut is in reverse edge order

    """
    ind = decode(individuo)
    dist = 0
    i1 = ind[0][0]
    a1 = edges[i1] if ind[1][0] == 0 else edges[i1][::-1]
    if a1 != (0.0, 0.0):
        dist += dist2pt(0.0, 0.0, *a1[0]) / mi
    dist += (dist2pt(*a1[0], *a1[1])) / pi
    for i in range(len(ind[0]) - 1):
        i1 = ind[0][i]
        i2 = ind[0][i + 1 if i + 1 < len(ind[0]) else 0]
        a1 = edges[i1] if ind[1][i] == 0 else edges[i1][::-1]
        a2 = edges[i2] if ind[1][i + 1 if i + 1 <
                                 len(ind[0]) else 0] == 0 else edges[i2][::-1]
        if a1[1] == a2[0]:
            dist += (dist2pt(*a2[0], *a2[1])) / pi
        else:
            dist += (dist2pt(*a1[1], *a2[0])) / mi + (
                dist2pt(*a2[0], *a2[1])) / pi
    iu = ind[0][-1]
    au = edges[iu] if ind[1][-1] == 0 else edges[iu][::-1]
    if au != (0.0, 0.0):
        dist += dist2pt(*au[1], 0.0, 0.0) / mi
    individuo.fitness.values = (dist, )
    return dist,


def main(P=1000, Pe=0.2, Pm=0.3, pe=0.7, NumGenWithoutConverge=100, file=None):
    """
    Execute Genetic Algorithm.

    args:
        P -> size of population
        Pe -> size of elite population
        Pm -> size of mutant population
        Pe -> elite allele inheritance probability
        NumGenWithoutConverge -> Number of generations without converge
        file -> if write results in file

    """
    tempo = timedelta(seconds=300)

    pop = toolbox.population(n=P)

    toolbox.register("mate", crossBRKGA, indpb=pe)

    tamElite = ceil(P * Pe)
    tamMutant = ceil(P * Pm)
    tamCrossover = P - tamElite - tamMutant

    gen, genMelhor = 0, 0

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Evaluate the entire population
    list(toolbox.map(toolbox.evaluate, pop))
    melhor = numpy.min([i.fitness.values for i in pop])
    logbook = tools.Logbook()
    p = stats.compile(pop)
    logbook.record(gen=0, **p)
    logbook.header = "gen", 'min', 'max', "avg", "std"
    gens, inds = [], []
    gens.append(gen)
    inds.append(melhor)
    print(logbook.stream, file=file)
    hora = datetime.now()
    while gen - genMelhor <= NumGenWithoutConverge:
        # Select the next generation individuals
        offspring = sorted(
            list(toolbox.map(toolbox.clone, pop)),
            key=lambda x: x.fitness,
            reverse=True
        )
        elite = offspring[:tamElite]
        cross = offspring[tamElite:tamCrossover]
        c = []
        # Apply crossover and mutation on the offspring
        for i in range(tamCrossover):
            e1 = random.choice(elite)
            c1 = random.choice(cross)
            ni = creator.Individual([[], []])
            ni[0] = toolbox.mate(e1[0], c1[0])
            ni[1] = toolbox.mate(e1[1], c1[1])
            c.append(ni)

        p = toolbox.population(n=tamMutant)
        c = elite + c + p
        offspring = c

        # Evaluate the individuals with an invalid fitness

        list(toolbox.map(toolbox.evaluate, offspring[tamElite:]))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        gen += 1
        minf = numpy.min([i.fitness.values for i in pop])
        men = False
        try:
            if minf < melhor:
                men = True
                melhor = minf
                genMelhor = gen
        except Exception:
            print(minf)

        p = stats.compile(pop)
        logbook.record(gen=gen, **p)
        if gen - genMelhor <= NumGenWithoutConverge and not men:
            print(logbook.stream)
        else:
            print(logbook.stream, file=file)
        hof.update(pop)
        gens.append(gen)
        inds.append(minf)

        if (datetime.now() - hora) > tempo:
            break
    return pop, stats, hof, gens, inds


def crossBRKGA(ind1, ind2, indpb):
    """."""
    return [ind1[i] if random.random() < indpb else ind2[i]
            for i in range(min(len(ind1), len(ind2)))]


files = [
    'albano',
    'blaz1',
    'blaz2',
    'blaz3',
    'dighe1',
    'dighe2',
    'fu',
    'instance_01_2pol',
    'instance_01_3pol',
    'instance_01_4pol',
    'instance_01_5pol',
    'instance_01_6pol',
    'instance_01_7pol',
    'instance_01_8pol',
    'instance_01_9pol',
    'instance_01_10pol',
    'instance_01_16pol',
    'instance_artificial_01_26pol_hole',
    'rco1',
    'rco2',
    'rco3',
    'shapes2',
    'shapes4',
    'spfc_instance',
    'trousers',
]

opcoes = {'pop': [100, 5000, 1000], 'elite': [.3, .2, .1], 'mut': [.1, .15, .2]}
op = []
for i in opcoes['pop']:
    for j in opcoes['elite']:
        for k in opcoes['mut']:
            op.append((i, j, k))
# toolbox of GA
toolbox = base.Toolbox()
# Class Fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Representation Individual
creator.create("Individual", list, fitness=creator.FitnessMin)
tipo = ['packing', 'separated']
if __name__ == "__main__":
    for t in tipo:
        for f in files:
            file = open(f"ejor/{t}/{f}.txt").read().strip().split('\n')
            edges = []
            if file:
                n = int(file.pop(0))
                for i in range(len(file)):
                    a = [float(j) for j in file[i].split()]
                    edges.append([(a[0], a[1]), (a[2], a[3])])
            # Generate Individual
            toolbox.register("indices", genIndividuoRK, edges)
            # initializ individual
            toolbox.register(
                "individual",
                tools.initIterate,
                creator.Individual,
                toolbox.indices
            )
            # Generate Population
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            # Objective Function
            toolbox.register("evaluate", evalCut)
            # function to execute map
            toolbox.register("map", map)

            hof = None
            for k in op:
                qtd = 10
                with open(f"resultados/brkga/{t}/{f}_[{k[0]},{k[1]},{k[2]}].txt", mode='w+') as \
                        file_write:
                    print("BRKGA:", file=file_write)
                    print(file=file_write)
                    for i in range(qtd):
                        print(f"Execução {i+1}:", file=file_write)
                        print(
                            f"Parametros: P={k[0]}, Pe={k[1]}, Pm={k[2]}, pe=0.7, Parada=150",
                            file=file_write
                        )
                        iteracao = None
                        with timeit(file_write=file_write):
                            iteracao = main(
                                P=k[0],
                                Pe=k[1],
                                Pm=k[2],
                                file=file_write
                            )
                        print("Individuo:", decode(iteracao[2][0]), file=file_write)
                        print("Fitness: ", iteracao[2][0].fitness.values[0], file=file_write)
                        print("Gens: ", iteracao[3], file=file_write)
                        print("Inds: ", iteracao[4], file=file_write)
                        print(file=file_write)
                        plotar(iteracao[2][0], f"{t}/{f}_[{k[0]}, {k[1]}, {k[2]}]" + str(i + 1))
                        fig1, f1_axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
                        gens, inds = iteracao[3], iteracao[4]
                        f1_axes.set_ylabel("Valor do Melhor Individuo")
                        f1_axes.set_xlabel("Gerações")
                        f1_axes.grid(True)
                        f1_axes.set_xlim(0, gens[-1])
                        f1_axes.set_ylim(inds[-1] - 10, inds[0] + 10)
                        f1_axes.plot(gens, inds, color='blue')
                        fig1.savefig(
                            f'melhora/brkga/{t}/' + f"{f}_[{k[0]}, {k[1]}, {k[2]}]" +
                            str(i + 1) + '.png',
                            dpi=300
                        )
                        plt.close()
