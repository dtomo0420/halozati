import matplotlib.pyplot as plt
import copy
import os
import time
from time import perf_counter


def read_input(filename):
    file_name = f"datasets/{filename}.txt"

    L = None
    array = []

    with open(file_name, 'r') as file:
        L = int(file.readline().strip())

        array_str = file.readline().strip()
        if array_str:
            array = eval(array_str)

    return L, array


# visualise the initial intervals and prepare for the following tasks
def vis_intervals(L, init_array, time_string):
    print('--- Initial begin ---')

    # x and y koordinates
    x = []
    y = []

    for i in range(0, len(init_array)):
        x_cur = []
        y_cur = []
        if init_array[i]:
            for j in range(0, 2):
                x_cur.append(init_array[i][j][0])
                y_cur.append(init_array[i][j][1])
                # visualise the current interval
            plt.plot(x_cur, y_cur, marker='o', color='red')
            # append current element to the x,y vectors
            x.append(x_cur)
            y.append(y_cur)

    # set up the details
    plt.xlim([0, L])
    plt.ylim([len(init_array), 0])
    plt.xlabel('To <--- L ---> to+L')
    plt.ylabel('Intervals')
    plt.title('Original intervals')
    plt.savefig(f'results/result_intervals{time_string}.png')
    plt.show()

    print('--- Initial end ---')
    return x, y


# implementation of the shirking process
def shirking(L, array, x_origin, y_origin):
    print('--- Shirking begin ---')

    # Set true to see details
    write = False
    x_sh_array = copy.deepcopy(x_origin)
    y_sh_array = copy.deepcopy(y_origin)
    if write:
        print('The initial array')
        print(x_origin)
        print(y_origin)
        print('___________')

    # examine all intervals
    for i in range(0, len(x_origin)):
        if write:
            print(str(i + 1) + '. interval')

        # in the case of the 1st interval choose automatic the left side
        # in any other cases choose the biggest shirking value
        if i == 0:
            dist = x_origin[i][0]
        else:
            distA = L - x_origin[i - 1][1] + x_origin[i][0]  # t0+A
            distB = L - x_origin[i][1]  # L-B

            dist = distA

            if dist < distB:
                dist = distB

        if write:
            print('The chosen distance: ', dist)

        x_sh_array[i][0] = x_origin[i][0] - dist
        x_sh_array[i][1] = x_origin[i][1] - dist

        # if the shirking overhangs the current row, it is moved to the previous row
        if x_sh_array[i][0] < 0:
            x_sh_array[i][0] = L - abs(x_origin[i][0])
            y_sh_array[i][0] = y_origin[i][0] - 1
        if x_sh_array[i][1] < 0:
            x_sh_array[i][1] = L - abs(x_origin[i][1])
            x_sh_array[i][1] = y_origin[i][1] - 1

        if write:
            print(x_sh_array)
            print(y_sh_array)
            print(array)

    # visualise the shirked intervals
    x = []
    y = []
    for i in range(0, len(x_sh_array)):
        x.append(x_sh_array[i][0])
        x.append(x_sh_array[i][1])
        y.append(y_sh_array[i][0])
        y.append(y_sh_array[i][1])

    plt.scatter(x, y)
    plt.xlim([0, L])
    plt.ylim([len(array), 0])
    plt.xlabel('To <--- L ---> to+L')
    plt.ylabel('Intervals')
    plt.title('A zsugoritas utani allapot')
    plt.close()

    print('--- Shirking end ---')
    return x, y


# create the patrol graph
def bag_graph(L, x_sh, y_sh):
    print('--- Bag begin ---')

    # create a vector what contain the labels
    namesparam = []
    for i in range(1, len(x_sh) // 2 + 1):
        namesparam.append('A' + str(i))
        namesparam.append('B' + str(i))

    names = namesparam
    xbag = []
    ybag = []
    namesbag = []

    # the first element
    xbag.append([x_sh[0]])
    ybag.append([y_sh[0]])
    namesbag.append([namesparam[0]])
    xparam = x_sh[1:]
    yparam = y_sh[1:]
    namesparam = namesparam[1:]

    rep = True
    while rep:
        num, rep = calc(L, xparam, yparam)
        xbag.append([xparam[0:num]])
        ybag.append([yparam[0:num]])
        namesbag.append([namesparam[0:num]])
        xparam = xparam[num:]
        yparam = yparam[num:]
        namesparam = namesparam[num:]

    print(namesbag)
    print('--- Bag end ---')
    return namesbag, names


def calc(L, xparam, yparam):
    dif = xparam[0]
    for i in range(1, len(xparam)):
        if xparam[i] - dif >= 0:
            xparam[i] = xparam[i] - dif
        else:
            xparam[i] = L + xparam[i] - dif
            yparam[i] -= 1

    sum = xparam[0]
    for i in range(1, len(xparam)):
        if yparam[i-1] == yparam[i]:
            sum -= xparam[i-1]
            sum += xparam[i]
        else:
            sum += xparam[i]
        if sum >= L:
            print(sum)
            return i, True

    return len(xparam), False


# create the patrol graph
def vis_patrol(L, array, bag, names, time_string):
    print('--- Patrol begin ---')
    x = []
    y = []
    x_c = []
    y_c = []

    routes = []

    # show the initial intervals
    for i in range(0, len(array)):
        x_cur = []
        y_cur = []
        if array[i]:
            for j in range(0, 2):
                x_cur.append(array[i][j][0])
                y_cur.append(array[i][j][1])
            plt.plot(x_cur, y_cur, marker='o', color='red')
            x.append(x_cur)
            y.append(y_cur)

    for i in range(0, len(x)):
        x_c.append(x[i][0])
        x_c.append(x[i][1])
        y_c.append(y[i][0])
        y_c.append(y[i][1])

    first_ready = False

    # create the shortcut graph
    for i in range(0, len(bag)):
        # skip the 1st bag
        if type(bag[i][0]) != str:
            for j in range(0, len(bag[i][0])):
                # check name
                if bag[i][0][j][0] == 'A' or first_ready:
                    if 0 <= j+1 < len(bag[i][0]):
                        print(bag[i][0][j+1], bag[i][0][j])
                        show_patrol(x_c, y_c, names, bag[i][0][j+1], bag[i][0][j])
                        routes.append([bag[i][0][j+1], bag[i][0][j]])
                    else:
                        print(bag[i][0][0], bag[i][0][j])
                        show_patrol(x_c, y_c, names, bag[i][0][0], bag[i][0][j])
                        routes.append([bag[i][0][0], bag[i][0][j]])

        if i >= 1:
            first_ready = True

    # set up the details
    plt.xlim([0, L])
    plt.ylim([len(array), 0])
    plt.xlabel('To <--- L ---> to+L')
    plt.ylabel('Intervals')
    plt.title('The patrol graph')
    plt.savefig(f'results/result_patrol{time_string}.png')
    plt.show()
    print('--- Patrol end ---')
    return routes, x_c, y_c


# visualise the patrol graph
def show_patrol(x_c, y_c, names, param1, param2):
    #print(x_c, y_c, names, param1, param2)
    vis = [[],[]]
    for i in range(0,len(names)):
        if names[i] == param1:
            #print(x_c[i], y_c[i])
            vis[0].append(x_c[i])
            vis[1].append(y_c[i])
        if names[i] == param2:
            #print(x_c[i], y_c[i])
            vis[0].append(x_c[i])
            vis[1].append(y_c[i])
    plt.plot(vis[0], vis[1], ls='dashed', color='blue')


# order the interval pairs
def order_routes(routes):
    for i in routes:
        if i[0][1] >= i[1][1]:
            tmp = i[0]
            i[0] = i[1]
            i[1] = tmp
        elif i[0][0] == 'B' and i[1][0] == 'B':
            tmp = i[0]
            i[0] = i[1]
            i[1] = tmp


# create the origin interval names
def name_prep(names):
    array = []
    for i in range(0, len(names), 2):
        array.append([names[i], names[i+1]])
    return array


# get the coordinates based of the name of the interval
def get_coordinates(x_c, y_c, names, name):
    tmp0 = 0
    tmp1 = 0
    for i in range(0, len(names)):
        if names[i] == name[0]:
            tmp0 = x_c[i], y_c[i]
        if names[i] == name[1]:
            tmp1 = x_c[i], y_c[i]
    return [tmp0[0], tmp1[0]],[tmp0[1], tmp1[1]]


# delete cycle routes from patrol routes
def remove_cycle_from_partrol(names_origin, partol_routes):
    print(partol_routes)
    print(names_origin)
    for i in names_origin:
        for j in partol_routes:
            if i == j:
                partol_routes.remove(j)
            elif j[0] == j[1]:
                partol_routes.remove(j)


# generating the return route
def reverse(bichromatic_routes):
    # if the length > 10 sure there is a circle in it
    # otherwise we just turn it over
    if len(bichromatic_routes) > 10:
        bichromatic_routes_tmp = bichromatic_routes[-7::-1]
    else:
        bichromatic_routes_tmp = bichromatic_routes[::-1]

    return bichromatic_routes + bichromatic_routes_tmp


# calculate one route in the bichromatic graph
def calculate_route(names_origin_tmp, partol_routes_tmp):
    used_intervals = []
    bichromatic_routes = []
    next_item = names_origin_tmp[0][0]

    while True:
        for i in names_origin_tmp:
            if i[0] == next_item:
                bichromatic_routes.append(i[0])
                bichromatic_routes.append(i[1])
                used_intervals.append(i)
                next_item = i[1]
                names_origin_tmp.remove(i)
                break

        if len(partol_routes_tmp) == 0:
            reverse_result = reverse(bichromatic_routes)
            return reverse_result, bichromatic_routes, used_intervals

        for j in partol_routes_tmp:
            if j[0] == next_item:
                bichromatic_routes.append(j[0])
                bichromatic_routes.append(j[1])
                next_item = j[1]
                partol_routes_tmp.remove(j)
                break

        if next_item[0] == 'B':
            reverse_result = reverse(bichromatic_routes)
            return reverse_result, bichromatic_routes, used_intervals


# creation of bichromatic cycle
def bichromatic_cycle(names_origin, partol_routes):
    print('--- Bichromatic begin ---')
    cycles = []
    bichromatic_cycles = []
    names_origin_tmp = copy.deepcopy(names_origin)
    partol_routes_tmp = copy.deepcopy(partol_routes)

    while True:
        reverse_result, bichromatic_routes, used_intervals = calculate_route(names_origin_tmp, partol_routes_tmp)

        cycles.append(bichromatic_routes)
        bichromatic_cycles.append(reverse_result)

        # remove the visited intervals from further calculations
        for i in used_intervals:
            if i in names_origin_tmp:
                names_origin_tmp.remove(i)

        if len(names_origin_tmp) == 0:
            break

    return bichromatic_cycles


# write the bichromatic cycles in textfile
def write_the_cycles(file, bichromatic_cycles):
    file.write(('-'*30+'\n')*2)
    file.write('Bichromatic cycles:' + '\n')
    for i in range(0,len(bichromatic_cycles)):
        file.write('The ' + str(i+1) + '. route' + '\n')
        for j in range(0, len(bichromatic_cycles[i]), 2):
            #print(str(bichromatic_cycles[i][j]) + '->' + str(bichromatic_cycles[i][j+1]))
            file.write('\t' + str(bichromatic_cycles[i][j]) + '->' + str(bichromatic_cycles[i][j+1]) + '\n')


# calculate and write in file the cost of the bichromatic routes
def calculate_distance_2(file, bichromatic_cycles, names, x_c):
    file.write(('-' * 30 + '\n') * 2)
    file.write('Cost of the routes:' + '\n')
    for i in range(0,len(bichromatic_cycles)):
        file.write('\t' + 'The ' + str(i+1) + '. route: ')

        route_cost = 0

        if len(bichromatic_cycles[i]) <= 4 :
            for j in range(0,len(bichromatic_cycles[i]), 2):
                #print(bichromatic_cycles[i][j], bichromatic_cycles[i][j + 1])
                A = names.index(bichromatic_cycles[i][j])
                B = names.index(bichromatic_cycles[i][j+1])

                route_cost = route_cost + abs(int(x_c[B]) - int(x_c[A]))

        else:
            for j in range(0,len(bichromatic_cycles[i]), 4):
                #print(bichromatic_cycles[i][j], bichromatic_cycles[i][j + 1])
                A = names.index(bichromatic_cycles[i][j])
                B = names.index(bichromatic_cycles[i][j+1])

                route_cost = route_cost + abs(int(x_c[B]) - int(x_c[A]))

        file.write(str(route_cost) + '\n')

def calculate_distance(file, bichromatic_cycles, names, x_c):
    file.write(('-' * 30 + '\n') * 2)
    file.write('Cost of the routes:' + '\n')
    for i in range(0,len(bichromatic_cycles)):
        file.write('\t' + 'The ' + str(i+1) + '. route: ')
        route_cost = 0

        for j in range(0,len(bichromatic_cycles[i]),2):
            for k in range(0,len(names),2):
                if (bichromatic_cycles[i][j],bichromatic_cycles[i][j+1]) == (names[k],names[k+1])\
                        or (bichromatic_cycles[i][j],bichromatic_cycles[i][j+1]) == (names[k+1],names[k]):

                    #print(bichromatic_cycles[i][j],bichromatic_cycles[i][j+1])

                    A = names.index(bichromatic_cycles[i][j])
                    B = names.index(bichromatic_cycles[i][j + 1])

                    route_cost = route_cost + abs(int(x_c[B]) - int(x_c[A]))

        #print(route_cost)
        file.write(str(route_cost) + '\n')



# calculate the cost of the tsp problem with one robot
def calc_tsp(file, names, x_c):
    route_cost = 0
    for i in range(0, len(x_c), 2):
        route_cost = route_cost + (x_c[i+1] - x_c[i])*2

    route_cost = route_cost - x_c[-1] + x_c[-2]

    file.write('\n' + 'Cost of the tsp: ' + str(route_cost) + '\n')
    #print(route_cost)

def calc_runtimes(time, time_tsp):
    file.write(('-' * 30 + '\n') * 2)
    file.write('The running times:' + '\n')
    file.write('\t' + f'based on the algorithm in the article: {time:.10f} sec' + '\n')
    file.write('\t' + f'based on the TSP algorithm: {time_tsp:.10f} sec' + '\n')
    pass


if __name__ == '__main__':
    named_tuple = time.localtime()
    time_string = time.strftime("_%Y_%m_%d__%H_%M_%S", named_tuple)

    file_source = 'results/result_file' + time_string + '.txt'

    if os.path.exists(file_source):
        os.remove(file_source)
    else:
        file = open(file_source, "x")

    file = open(file_source, "w")

    L, array = read_input('origin')
    #L, array = read_input('origin2')
    #L, array = read_input('origin3')

    #L, array = read_input('bad')

    start = time.time()

    x_origin, y_origin = vis_intervals(L, array, time_string)

    x_sh, y_sh = shirking(L, array, x_origin, y_origin)

    bag, names = bag_graph(L, x_sh, y_sh)

    partol_routes, x_c, y_c = vis_patrol(L, array, bag, names, time_string)

    #print('Partrol routres:' + '\n' + str(partol_routes))
    names_origin = name_prep(names)

    order_routes(partol_routes)
    #print('The ordered partrol routres:' + '\n' + str(partol_routes))

    remove_cycle_from_partrol(names_origin, partol_routes)

    bichromatic_cycles = bichromatic_cycle(names_origin, partol_routes)
    print('The bichromatic cycles:' + '\n' + str(bichromatic_cycles))

    write_the_cycles(file, bichromatic_cycles)

    end = time.time()

    calculate_distance(file, bichromatic_cycles, names, x_c)

    start_tsp = perf_counter()

    calc_tsp(file, names, x_c)

    end_tsp = perf_counter()

    calc_runtimes(end - start, end_tsp - start_tsp)

    #calculate_distance_2(file, bichromatic_cycles, names, x_c)

