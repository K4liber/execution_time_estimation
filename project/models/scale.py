from sklearn.preprocessing import StandardScaler

scale_x = None
scale_y = None


def init_scale(x: any, y: any):
    global scale_x
    global scale_y
    scale_x = StandardScaler().fit(x)
    scale_y = StandardScaler().fit(y)


def dismiss_scale():
    global scale_x
    global scale_y
    scale_x = None
    scale_y = None


def transform_x(x: any) -> any:
    if scale_x is not None:
        return scale_x.transform(x)
    else:
        return x


def transform_y(y: any) -> any:
    if scale_y is not None:
        return scale_y.transform(y)
    else:
        return y


def inverse_transform_y(y: any) -> any:
    if scale_y is not None:
        return scale_y.inverse_transform(y)
    else:
        return y
