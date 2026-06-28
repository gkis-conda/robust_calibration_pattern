import numpy as np

def menger_curvature_loss(params, points_2d, lines, weights, center):
    f, k1 = params
    if f < 100: return 1.e10 # Физический барьер для оптимизатора
    
    # 1. Выпрямляем все точки кадра по текущей модели дисторсии
    from __main__ import undistort_points
    u_points = undistort_points(points_2d, f, k1, center)
    
    total_loss = 0.0
    eps = 1.e-6 # Защита от деления на ноль (если точки совпали)
    
    for line in lines:
        n_pts = len(line)
        if n_pts < 3: 
            continue
            
        # Скользящее окно по триплетам вдоль всей дуги
        for i in range(n_pts - 2):
            p1 = u_points[line[i]]
            p2 = u_points[line[i+1]]
            p3 = u_points[line[i+2]]
            
            # Векторы между точками триплета
            v1 = p2 - p1
            v2 = p3 - p2
            v3 = p3 - p1
            
            # Площадь параллелограмма (числитель формулы Менгера)
            cross_prod = v1[0]*v2[1] - v1[1]*v2[0]
            
            # Длины всех трех сторон локального треугольника
            len_v1 = np.sqrt(v1[0]**2 + v1[1]**2) + eps
            len_v2 = np.sqrt(v2[0]**2 + v2[1]**2) + eps
            len_v3 = np.sqrt(v3[0]**2 + v3[1]**2) + eps
            
            # Строгая кривизна Менгера (1/R)
            kappa = (2.0 * cross_prod) / (len_v1 * len_v2 * len_v3)
            
            # Индекс центральной точки триплета для взвешивания
            pt_id = line[i+1]
            
            # Накапливаем квадрат кривизны, умноженный на инцидентность точки.
            # Если линия идеально прямая, kappa = 0.
            total_loss += weights[pt_id] * (kappa ** 2)
            
    return total_loss
