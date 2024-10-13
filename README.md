## Запуск программы
python .\cgm.py 0.1 4 9 \
где первое значение отвечает за точность \
второе и третье это начальные значения функции 




## Минимизируемая функция

Рассматриваемая функция: 
$$f(x, y) =  ((x-5)^2)*(y-10)^2 +(x-5)^2 + (y-10)^2 + 1 $$ \
Она имеет глобальный минимум в точке $(x, y) = (5, 10)$, где $f(x, y) = 1$.
## Выводы 
Программа выдаёт минимальное значение функции; точку, в которой оно достигается; количество итераций; количество вычисленных значений. 

## Метод сопряженных градиентов

Направление спуска в определяется вектором $p$:  $$p^{i} = \gamma^{i-1} p^{i-1} + \alpha^i; \text{ } i \in \mathbb{N}; \text{ } p^1 = \alpha^1,$$

где $\alpha^i = -grad f(x^{i-1})$ — антиградиент целевой функции в точке $x^{i-1}$, $i$ — номер текущей итерации.\

Значение $\gamma^i$ на каждой итерации вычисляется по формуле:

$$\gamma^i = - \frac{(H(x^i)p^i, \alpha^{i+1})}{(H(x^i)p^i, p^i)}; \text{ } i \in \mathbb{N},$$

где $H(x^i)$ — матрица Гессе целевой функции в точке $x^i$.

С целью ослабить влияние погрешностей, в данном методе применяется процедура "обновления" алгоритма. Суть процедуры заключается в том, что периодически через заданное число итераций принимаем $\gamma^i = 0$. Соответсвующий номер итерации называется моментом обновления алгоритма. Множество таких моментов имеет вид:

$$n, 2n, ..., mn; \text{ } m \in \mathbb{N}.$$ 

Элементы релаксационной последовательности $x^i$ строим при помощи рекурентного соотношения вида:

$$x^i = x^{i - 1} + \xi^i p^i; \text{ } i \in \mathbb{N}; \text{ } \xi > 0.$$

Значение $\xi^i$ на каждой итерации выбирается путем минимизации функции:

$$\psi^i(\xi) = f(x^{i-1} + \xi p^i); \text{ } i \in \mathbb{N}; \text{ } \xi > 0.$$

В качестве критерия останова выбрано условие:

$$\left| \alpha^i \right| < \epsilon; \text{ } i \in \mathbb{N}; \text{ } \epsilon > 0,$$

где $\epsilon$ — заданная точность.
