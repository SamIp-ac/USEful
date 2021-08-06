import pulp
import pandas as pd

A = pulp.LpVariable('A', lowBound=0)
B = pulp.LpVariable('B', lowBound=0, upBound=10000)

profit = pulp.LpProblem('Max profit', pulp.LpMaximize)
# Objective function
profit += 5*A+4*B, 'Objective function'
profit += 1/2*A+B <= 500, 'Constraint 1'
profit += A+1/4*B <= 400, 'Constraint 2'

profit.solve()

print('The Decision variables are : ')
for variable in profit.variables():
    print(variable.name, '=', variable.varValue)

print('\nMax profit $', pulp.value(profit.objective))

o = [{'name': name, 'shadow price': c.pi, 'slack': c.slack}
     for name, c in profit.constraints.items()]
print(pd.DataFrame(o))
