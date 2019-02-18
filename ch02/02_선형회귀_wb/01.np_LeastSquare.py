import numpy as np

x = [2, 4, 6, 8]
y = [81, 93, 91, 97]
#print(len(x), len(y))

mx = np.mean(x)
my = np.mean(y)
print("mean_x:%f, mean_y:%f" % (mx, my))

#tmp = [ (i -mx)**2 for i in x ]
#print(tmp)

#divisor = sum([ (mx -i)**2 for i in x])
divisor = sum([ (i -mx)**2 for i in x])

def func_nu(x, mx, y, my):
    nu = 0
    for i in range(len(x)):
        nu += (x[i] - mx) * ( y[i] - my)
    return nu
numerator = func_nu(x, mx, y, my)

print("divisor:%f, numerator:%f" % (divisor, numerator))

a = numerator / divisor
b = my - (mx * a)
print("a : ", a)
print("b : ", b)


