矩阵分解包里面 mf-ips 被我改了, 具体就是把propensity 用成 逻辑回归返回的了

然后snips 是单独写了一份,   可以在main.py 调包的部分选择调用matrix-factorization里面的snips的实现
或者调snips里面snips的实现.

indicator_propensity就是 调用逻辑回归计算propensity, 我把整个放到了一个方法里, 可以在其他文件里调用,
会返回propensity tensor
