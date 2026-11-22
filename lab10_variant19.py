"""Лабораторная работа № 10. Вариант 19. Ситникова Ольга Алексеевна."""
import heapq  # приоритетная очередь (двоичная куча)
import platform  # сведения о компьютере
import random  # случайные графы
import time  # замер времени
from collections import deque  # очередь для алгоритма Кана
import matplotlib.pyplot as plt  # графики

INF = float("inf")  # «бесконечность» для начальных оценок расстояний
SIZES = [1000, 2500, 5000]  # размеры графа n по варианту 19 (разреженный, m ≈ 5n, веса 1…40)


def dijkstra_array(adj, s):  # Дейкстра с линейным поиском минимума, Θ(n²)
    n = len(adj)  # число вершин
    dist, parent, used = [INF] * n, [-1] * n, [False] * n  # расстояния, предки и отметки «просмотрена»
    dist[s] = 0  # расстояние до источника
    for _ in range(n):  # n шагов
        v, best = -1, INF  # вершина с минимальной оценкой
        for i in range(n):  # линейный поиск минимума среди непросмотренных
            if not used[i] and dist[i] < best:  # нашли оценку меньше текущей
                v, best = i, dist[i]  # запоминаем вершину и оценку
        if v == -1:  # достижимые вершины закончились
            break  # выходим
        used[v] = True  # вершина просмотрена: её оценка окончательна
        for u, w in adj[v]:  # релаксация исходящих рёбер
            if dist[v] + w < dist[u]:  # путь через v короче
                dist[u], parent[u] = dist[v] + w, v  # обновляем оценку и предка
    return dist, parent  # расстояния и предки


def dijkstra_heap(adj, s):  # Дейкстра с приоритетной очередью, O(m log n)
    n = len(adj)  # число вершин
    dist, parent = [INF] * n, [-1] * n  # расстояния и предки
    dist[s] = 0  # расстояние до источника
    heap = [(0, s)]  # куча пар (оценка, вершина)
    while heap:  # пока куча не пуста
        d, v = heapq.heappop(heap)  # вершина с наименьшей оценкой
        if d > dist[v]:  # запись устарела
            continue  # пропускаем
        for u, w in adj[v]:  # релаксация исходящих рёбер
            if d + w < dist[u]:  # путь через v короче
                dist[u], parent[u] = d + w, v  # обновляем оценку и предка
                heapq.heappush(heap, (dist[u], u))  # кладём новую оценку в кучу
    return dist, parent  # расстояния и предки


def restore_path(parent, target):  # восстановление пути по массиву предков
    path = []  # вершины пути в обратном порядке
    while target != -1:  # идём от цели к источнику
        path.append(target)  # добавляем вершину
        target = parent[target]  # переходим к предку
    return path[::-1]  # путь от источника к цели


def generate_graph(n, m, max_weight=40, seed=42):  # случайный ориентированный граф: n вершин, около m рёбер
    rng = random.Random(seed)  # генератор с фиксированным seed
    adj = [[] for _ in range(n)]  # списки смежности с весами
    for i in range(1, n):  # остовное дерево гарантирует достижимость всех вершин из 0
        adj[rng.randrange(i)].append((i, rng.randint(1, max_weight)))  # ребро от случайной предыдущей вершины
    for _ in range(m - (n - 1)):  # остальные рёбра случайны
        u, v = rng.randrange(n), rng.randrange(n)  # случайная пара вершин
        if u != v:  # петли не допускаются
            adj[u].append((v, rng.randint(1, max_weight)))  # ребро со случайным весом
    return adj  # граф


def topological_order(tasks):  # топологическая сортировка (алгоритм Кана); tasks: код -> (длительность, предшественники)
    indeg = {k: len(v[1]) for k, v in tasks.items()}  # входящие степени
    succ = {k: [] for k in tasks}  # последователи каждой работы
    for k, (_, preds) in tasks.items():  # строим списки последователей
        for p in preds:  # для каждого предшественника
            succ[p].append(k)  # k следует за p
    queue = deque(sorted(k for k, d in indeg.items() if d == 0))  # работы без предшественников
    order = []  # топологический порядок
    while queue:  # пока очередь не пуста
        v = queue.popleft()  # берём работу
        order.append(v)  # добавляем в порядок
        for u in succ[v]:  # уменьшаем степени последователей
            indeg[u] -= 1  # одним предшественником меньше
            if indeg[u] == 0:  # все предшественники выполнены
                queue.append(u)  # работа готова к началу
    if len(order) != len(tasks):  # в порядок вошли не все работы
        raise ValueError("Сетевой график содержит цикл")  # цикл — ошибка исходных данных
    return order, succ  # порядок и последователи


def critical_path(tasks):  # метод критического пути: сроки, резервы, критические работы
    order, succ = topological_order(tasks)  # порядок работ
    ES, EF = {}, {}  # ранний старт и раннее окончание
    for v in order:  # прямой проход
        ES[v] = max((EF[p] for p in tasks[v][1]), default=0)  # старт после всех предшественников
        EF[v] = ES[v] + tasks[v][0]  # окончание = старт + длительность
    T = max(EF.values())  # длительность проекта
    LS, LF = {}, {}  # поздний старт и позднее окончание
    for v in reversed(order):  # обратный проход
        LF[v] = min((LS[u] for u in succ[v]), default=T)  # окончание не позже старта последователей
        LS[v] = LF[v] - tasks[v][0]  # поздний старт
    R = {v: LS[v] - ES[v] for v in tasks}  # полный резерв времени
    return T, ES, EF, LS, LF, R, [v for v in order if R[v] == 0]  # критические работы имеют нулевой резерв


def sensitivity(tasks):  # К5: изменение срока при +1 день и наименьшее увеличение, меняющее срок проекта
    T0, res = critical_path(tasks)[0], {}  # исходный срок и результат
    for v, (t, preds) in tasks.items():  # перебираем работы
        delta = 1  # пробное увеличение длительности
        while critical_path({**tasks, v: (t + delta, preds)})[0] == T0:  # срок не изменился
            delta += 1  # увеличиваем ещё на день
        res[v] = (critical_path({**tasks, v: (t + 1, preds)})[0] - T0, delta)  # (рост срока при +1 день, порог)
    return res  # результаты по работам


def best_time(func, repeats):  # лучшее время func из нескольких повторов, с
    times = []  # времена повторов
    for _ in range(repeats):  # повторы
        t0 = time.perf_counter()  # начало замера
        func()  # запуск
        times.append(time.perf_counter() - t0)  # длительность
    return min(times)  # наименее искажённое помехами значение


if __name__ == "__main__":  # запуск только при прямом старте файла
    print(platform.platform(), "|", platform.processor() or platform.machine(), "| Python", platform.python_version())  # система
    small = [[(1, 4), (2, 2)], [(3, 5)], [(1, 1), (3, 8), (4, 10)], [(4, 2), (5, 6)], [(5, 3), (6, 5)], [(6, 4)], []]  # тестовый граф: 7 вершин (1…7), 11 рёбер
    d1, p1 = dijkstra_array(small, 0)  # реализация с массивом
    d2, p2 = dijkstra_heap(small, 0)  # реализация с кучей
    assert d1 == d2 and p1 == p2  # обе реализации совпадают
    print("\nТестовый граф, расстояния от вершины 1 и кратчайшие пути:")  # заголовок
    for v in range(7):  # по всем вершинам
        print(f"  до {v + 1}: {d1[v]:>2}, путь {' -> '.join(str(x + 1) for x in restore_path(p1, v))}")  # результат
    print("\n     n       m | массив, с |   куча, с | отношение")  # заголовок таблицы
    rows = []  # результаты замеров
    for n in SIZES:  # перебираем размеры
        adj = generate_graph(n, 5 * n)  # разреженный граф m ≈ 5n
        m = sum(len(a) for a in adj)  # фактическое число рёбер
        assert dijkstra_array(adj, 0)[0] == dijkstra_heap(adj, 0)[0]  # реализации дают одинаковые расстояния
        t_arr = best_time(lambda: dijkstra_array(adj, 0), 3 if n <= 2500 else 2)  # время реализации с массивом
        t_heap = best_time(lambda: dijkstra_heap(adj, 0), 15)  # время реализации с кучей
        rows.append((n, m, t_arr, t_heap))  # сохраняем строку
        print(f"{n:>6} {m:>7} | {t_arr:>9.4f} | {t_heap:>9.4f} | {t_arr / t_heap:>9.1f}")  # строка таблицы
    adj = generate_graph(1000, 5000)  # граф для примера восстановления пути
    dist, parent = dijkstra_heap(adj, 0)  # расстояния от вершины 0
    far = max(range(1000), key=lambda v: dist[v] if dist[v] < INF else -1)  # самая удалённая вершина
    print(f"\nn = 1000: самая удалённая вершина {far}, расстояние {dist[far]}, путь: {' -> '.join(map(str, restore_path(parent, far)))}")  # путь
    tasks = {  # сетевой график проекта из 14 работ: код -> (длительность, предшественники)
        "A": (5, []), "B": (7, ["A"]), "C": (4, ["B"]), "D": (6, ["B"]), "E": (10, ["C"]), "F": (9, ["D"]),  # работы A–F
        "G": (3, ["E", "F"]), "H": (4, ["C", "D"]), "I": (6, ["G"]), "J": (2, ["H", "I"]),  # работы G–J
        "K": (3, ["B"]), "L": (5, ["C", "K"]), "M": (4, ["I", "L"]), "N": (2, ["J", "M"]),  # работы K–N
    }
    T, ES, EF, LS, LF, R, crit = critical_path(tasks)  # расчёт критического пути
    sens = sensitivity(tasks)  # анализ чувствительности (К5)
    print(f"\nДлительность проекта: {T} дн.")  # срок проекта
    print("Работа  t  ES  EF  LS  LF  Резерв  Крит.  +1 день -> срок  Порог роста срока")  # заголовок таблицы
    for v in sorted(tasks):  # по работам
        print(f"{v:>6} {tasks[v][0]:>2} {ES[v]:>3} {EF[v]:>3} {LS[v]:>3} {LF[v]:>3} {R[v]:>7} {'да' if R[v] == 0 else '':>6} {'+' + str(sens[v][0]):>9}      {sens[v][1]:>10}")  # строка
    print("Критический путь:", " -> ".join(crit))  # критические работы
    cyc = {**tasks, "A": (5, ["N"])}  # график с искусственным циклом
    try:  # проверка обнаружения цикла
        critical_path(cyc)  # расчёт должен завершиться ошибкой
    except ValueError as e:  # ожидаемая ошибка
        print("Проверка цикла:", e)  # сообщение об ошибке
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))  # два графика рядом
    ns = [r[0] for r in rows]  # размеры n
    ax1.plot(ns, [r[2] for r in rows], "o-", label="Массив, Θ(n²)")  # время реализации с массивом
    ax1.plot(ns, [r[3] for r in rows], "s-", label="Куча, O(m log n)")  # время реализации с кучей
    ax1.set(xlabel="n", ylabel="Время, с", title="Алгоритм Дейкстры (m ≈ 5n)", xscale="log", yscale="log")  # оформление
    ax1.legend()  # легенда
    codes = sorted(tasks)  # коды работ
    ax2.bar(codes, [R[v] for v in codes], color=["#d9534f" if R[v] == 0 else "#5b9bd5" for v in codes])  # резервы: критические красным
    for i, v in enumerate(codes):  # подписываем значение резерва над каждым столбцом
        ax2.text(i, R[v] + 0.2, str(R[v]), ha="center", color="#d9534f" if R[v] == 0 else "black")  # у критических работ — «0»
    ax2.set(xlabel="Работа", ylabel="Полный резерв, дн.", title="Резервы времени (у критических работ — 0)")  # оформление
    for ax in (ax1, ax2):  # общее оформление
        ax.grid(True, alpha=0.3)  # сетка
    plt.tight_layout()  # убираем наложение подписей
    plt.savefig("lab10_graphs.png", dpi=150)  # сохраняем график для отчёта
    plt.show()  # показываем график на экране
