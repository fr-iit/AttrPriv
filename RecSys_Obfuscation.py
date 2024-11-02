"""
PerBlur is extension of previous work proposed by Windenberg et al., (BlurMe: Inferring and Obfuscating
User Gender Based on Ratings ) and Strucks et al., (BlurM(or)e: Revisiting Gender Obfuscation
in the User-Item Matrix)

This code is extending previous github repository done by Christopher Strucks (Github Link: https://github.com/STrucks/BlurMore)

In PerBlur you need to :
    + Generate json file: "Confidence score" from imputation/knn/few_observed_entries
    + You will read the json file
"""

from sklearn.model_selection import GridSearchCV

import RecSys_DataLoader as DL
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import pandas as pd
import json
import math

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def blurMe():

    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    p = 0.01
    notice_factor = 2 #determines the maximum number of ratings allowed for obfuscation.
    dataset = ['100k', '1m', 'yahoo'][0]

    if dataset == '100k':
        X = DL.load_user_item_matrix_100k()
        T = DL.load_gender_vector_100k()
        X_filled = DL.load_user_item_matrix_100k_Impute()
        L_m = [751, 327, 24, 688, 472, 888, 317, 948, 315, 178, 305, 179, 359, 186, 10, 883, 294, 342, 5, 471, 1048, 1101, 879, 449, 324, 150, 357, 411, 1265, 136, 524, 678, 202, 692, 100, 303, 264, 339, 1010, 276, 218, 995, 240, 433, 271, 203, 544, 520, 410, 222, 1028, 924, 823, 322, 462, 265, 750, 23, 281, 137, 333, 736, 164, 358, 301, 479, 647, 168, 144, 94, 687, 60, 70, 175, 6, 354, 344, 1115, 597, 596, 42, 14, 521, 184, 393, 188, 450, 1047, 682, 7, 760, 737, 270, 1008, 260, 295, 1051, 289, 430, 235, 258, 330, 56, 347, 52, 205, 371, 1024, 8, 369, 116, 79, 831, 694, 287, 435, 495, 242, 33, 201, 230, 506, 979, 620, 355, 181, 717, 936, 500, 893, 489, 1316, 855, 2, 180, 871, 755, 523, 477, 227, 412, 183, 657, 510, 77, 474, 1105, 67, 705, 362, 1133, 511, 546, 768, 96, 457, 108, 1137, 352, 341, 825, 165, 584, 505, 908, 404, 679, 15, 513, 117, 490, 665, 28, 338, 59, 32, 1014, 989, 351, 475, 57, 864, 969, 177, 316, 463, 134, 703, 306, 378, 105, 99, 229, 484, 651, 157, 232, 114, 161, 395, 754, 931, 591, 408, 685, 97, 554, 468, 455, 640, 473, 683, 300, 31, 1060, 650, 72, 191, 259, 1280, 199, 826, 747, 68, 1079, 887, 578, 329, 274, 174, 428, 905, 780, 753, 396, 4, 277, 71, 519, 1062, 189, 325, 502, 233, 1022, 880, 1063, 197, 813, 16, 331, 208, 162, 11, 963, 501, 820, 930, 896, 318, 1142, 1194, 425, 171, 282, 496, 26, 215, 573, 527, 730, 49, 693, 517, 336, 417, 207, 900, 299, 226, 606, 268, 192, 407, 62, 636, 480, 1016, 241, 945, 343, 603, 231, 469, 515, 492, 664, 972, 642, 781, 984, 187, 149, 257, 40, 141, 978, 44, 845, 85, 244, 512, 182, 1021, 756, 1, 778, 644, 1050, 185, 530, 81, 497, 3, 335, 923, 509, 418, 724, 566, 221, 570, 655, 413, 1311, 340, 583, 537, 635, 686, 176, 296, 926, 561, 101, 173, 862, 680, 652]
        L_f = [332, 321, 877, 906, 827, 882, 337, 278, 872, 304, 87, 93, 1038, 292, 629, 727, 220, 898, 107, 143, 516, 819, 382, 499, 876, 45, 1600, 111, 416, 132, 752, 125, 689, 604, 628, 310, 246, 419, 148, 659, 881, 248, 899, 707, 20, 129, 272, 690, 350, 498, 19, 216, 713, 429, 253, 372, 311, 251, 288, 1023, 392, 1328, 131, 320, 83, 895, 912, 445, 1094, 432, 109, 156, 532, 588, 486, 653, 243, 319, 518, 269, 298, 447, 290, 1386, 988, 423, 723, 279, 696, 568, 919, 1296, 266, 645, 937, 595, 482, 748, 348, 749, 815, 451, 154, 684, 993, 904, 204, 198, 403, 525, 553, 671, 507, 1119, 1036, 65, 420, 169, 9, 427, 832, 894, 729, 654, 1128, 987, 1281, 401, 1089, 942, 190, 406, 170, 402, 878, 291, 710, 297, 721, 151, 367, 954, 856, 64, 312, 385, 581, 370, 39, 648, 691, 746, 89, 1033, 745, 194, 662, 145, 539, 98, 147, 739, 159, 90, 236, 55, 120, 448, 200, 409, 623, 118, 731, 365, 238, 124, 146, 716, 1026, 847, 261, 153, 262, 130, 127, 866, 58, 356, 195, 1086, 1190, 1041, 1073, 889, 405, 950, 891, 611, 582, 66, 1035, 346, 1394, 172, 909, 1013, 785, 925, 488, 704, 1315, 708, 1431, 309, 121, 239, 902, 213, 446, 1012, 1054, 873, 193, 478, 829, 916, 353, 349, 133, 249, 550, 638, 452, 1313, 466, 529, 458, 792, 313, 625, 668, 1061, 742, 1065, 763, 73, 846, 1090, 863, 1620, 890, 1176, 1017, 237, 1294, 245, 508, 387, 53, 514, 422, 1068, 1527, 939, 1232, 1011, 631, 381, 956, 875, 459, 607, 1442, 155, 697, 1066, 1285, 293, 88, 1221, 1109, 675, 254, 209, 649, 140, 48, 613, 962, 1157, 991, 1083, 720, 535, 901, 436, 286, 559, 1085, 892, 102, 211, 285, 1383, 758, 1234, 674, 1163, 283, 637, 735, 885, 630, 709, 938, 38, 142, 219, 953, 275, 1059, 676, 210, 47, 63, 255, 494, 744, 250, 1114, 672, 308, 632, 17, 1203, 762, 1522, 538, 491, 307, 669, 775, 1135, 217, 841, 949, 766, 966, 314, 812, 821, 574, 1039, 1388, 51, 579, 252, 50, 1009, 1040, 1147, 224, 212, 1483, 1278, 122, 934, 702, 443, 86, 1444, 69, 400, 975, 35, 549, 928, 531, 622, 614, 1020, 658, 740, 1468, 22, 1007, 470, 1269, 633, 1074, 1299, 1095, 1148, 280, 1136, 92, 167, 434, 284, 961, 1084, 126, 619, 974, 196, 485, 1152, 673, 627, 345, 29, 605, 929, 952, 714, 1053, 526, 123, 476, 660, 955, 380, 1503, 163, 493, 1197, 431, 504, 886, 1037, 13, 794, 273, 844, 1032, 1025, 106, 91, 533, 421, 699, 869, 78, 1243, 481, 661, 82, 732]

    elif dataset == '1m':
        X = DL.load_user_item_matrix_1m()
        T = DL.load_gender_vector_1m()
        X_filled = DL.load_user_item_matrix_1m_Impute()
        L_m = [589.0, 1233.0, 2706.0, 1090.0, 2959.0, 1250.0, 2427.0, 2490.0, 1208.0, 1266.0, 3654.0, 1748.0, 1262.0, 1396.0, 1374.0, 2700.0, 1265.0, 1089.0, 1222.0, 231.0, 2770.0, 1676.0, 2890.0, 1228.0, 1136.0, 3360.0, 3298.0, 1663.0, 3811.0, 2011.0, 1261.0, 233.0, 3361.0, 2366.0, 1127.0, 1276.0, 3555.0, 1214.0, 3929.0, 299.0, 1304.0, 3468.0, 1095.0, 150.0, 1213.0, 750.0, 3082.0, 6.0, 111.0, 3745.0, 349.0, 541.0, 2791.0, 785.0, 1060.0, 1294.0, 1302.0, 1256.0, 1292.0, 2948.0, 3683.0, 3030.0, 3836.0, 913.0, 2150.0, 32.0, 2826.0, 2721.0, 590.0, 3623.0, 2997.0, 3868.0, 3147.0, 1610.0, 3508.0, 2046.0, 21.0, 1249.0, 10.0, 1283.0, 3760.0, 2712.0, 3617.0, 3552.0, 3256.0, 1079.0, 3053.0, 1517.0, 2662.0, 1953.0, 2670.0, 3578.0, 2371.0, 3334.0, 2502.0, 2278.0, 364.0, 3462.0, 2401.0, 3163.0, 2311.0, 852.0, 2916.0, 1378.0, 3384.0, 524.0, 70.0, 370.0, 3035.0, 3513.0, 2917.0, 3697.0, 24.0, 1957.0, 3494.0, 1912.0, 3752.0, 2013.0, 3452.0, 3928.0, 2987.0, 431.0, 2759.0, 1387.0, 1882.0, 3638.0, 1288.0, 2867.0, 2728.0, 2433.0, 161.0, 3386.0, 517.0, 741.0, 1287.0, 1231.0, 3062.0, 2288.0, 3753.0, 529.0, 3793.0, 3052.0, 2447.0, 1320.0, 3819.0, 1303.0, 922.0, 3022.0, 260.0, 858.0, 493.0, 3006.0, 480.0, 2410.0, 333.0, 1178.0, 3814.0, 2702.0, 1203.0, 2922.0, 1625.0, 3366.0, 3213.0, 2188.0, 2628.0, 3358.0, 2648.0, 3788.0, 953.0, 999.0, 3754.0, 3910.0, 3016.0, 3863.0, 303.0, 3263.0, 1080.0, 786.0, 3764.0, 2105.0, 3543.0, 2607.0, 3681.0, 592.0, 145.0, 2303.0, 1682.0, 1019.0, 3646.0, 1544.0, 235.0, 908.0, 3615.0, 2792.0, 2167.0, 2455.0, 1587.0, 1227.0, 2901.0, 2687.0, 1883.0, 1210.0, 1201.0, 3169.0, 3098.0, 3688.0, 2409.0, 3198.0, 610.0, 1923.0, 1982.0, 165.0, 2403.0, 784.0, 2871.0, 2889.0, 628.0, 2300.0, 417.0, 3671.0, 3100.0, 3914.0, 3608.0, 3152.0, 3429.0, 1794.0, 952.0, 1391.0, 2518.0, 410.0, 3535.0, 2333.0, 1713.0, 2605.0, 707.0, 2795.0, 1965.0, 373.0, 3916.0, 556.0, 3703.0, 95.0, 466.0, 3066.0, 3177.0, 2088.0, 1476.0, 163.0, 3422.0, 58.0, 1244.0, 1689.0, 2002.0, 1711.0, 2259.0, 3524.0, 1371.0, 3104.0, 1693.0, 965.0, 1732.0, 2600.0, 3424.0, 3755.0, 2450.0, 3826.0, 3801.0, 3927.0, 1298.0, 2118.0, 112.0, 2478.0, 471.0, 1673.0, 1246.0, 2734.0, 2529.0, 2806.0, 1948.0, 2093.0, 45.0, 648.0, 3504.0, 2968.0, 1722.0, 1963.0, 2840.0, 1747.0, 1348.0, 3871.0, 3175.0, 2360.0, 1092.0, 3190.0, 1405.0, 367.0, 3248.0, 1702.0, 1734.0, 2644.0, 1597.0, 1401.0, 1416.0, 107.0, 1379.0, 2764.0, 2116.0, 1036.0, 60.0, 2115.0, 1876.0, 1254.0, 2243.0, 2606.0, 3925.0, 3087.0, 1627.0, 3770.0, 3678.0, 3113.0, 3036.0, 3525.0, 1584.0, 2236.0, 3267.0, 954.0, 1205.0, 2470.0, 2686.0, 3397.0, 2015.0, 1377.0, 3740.0, 1594.0, 2456.0, 2038.0, 891.0, 1342.0, 1966.0, 2808.0, 3324.0, 3794.0, 2467.0, 3420.0, 3773.0, 1927.0, 2231.0, 3742.0, 1960.0, 1542.0, 2672.0, 1376.0, 3174.0, 1248.0, 225.0, 1267.0, 3203.0, 1025.0, 2769.0, 1973.0, 2541.0, 3593.0, 2058.0, 3273.0, 154.0, 1179.0, 2009.0, 2423.0, 2676.0, 2793.0, 3505.0, 1920.0, 3357.0, 2580.0, 2542.0, 1701.0, 3252.0, 440.0, 540.0, 1885.0, 2384.0, 1414.0, 1251.0, 1187.0, 2841.0, 2287.0, 2004.0, 1257.0, 1358.0, 2253.0, 3918.0, 2976.0, 1100.0, 2140.0, 2092.0, 2772.0, 3500.0, 1196.0, 3728.0, 555.0, 3564.0, 3099.0, 2863.0, 2492.0, 13.0, 2378.0, 3271.0, 3946.0, 1017.0, 3189.0, 3908.0, 1238.0, 3551.0, 800.0, 1193.0, 3254.0, 3614.0, 448.0, 1779.0, 3477.0, 1388.0, 748.0, 1411.0, 3948.0, 1057.0, 2877.0, 2633.0, 3078.0, 2289.0, 514.0, 3831.0, 535.0, 361.0, 290.0, 1408.0, 1356.0, 2522.0, 2321.0, 1395.0, 1103.0, 2861.0, 1974.0, 2497.0, 1633.0, 2530.0, 1931.0, 125.0, 1735.0, 3159.0, 892.0, 2828.0, 523.0, 3148.0, 296.0, 2882.0, 1639.0, 1665.0, 3834.0, 534.0, 2942.0, 1247.0, 861.0, 2107.0, 3469.0, 1970.0, 3307.0, 432.0, 3879.0, 3930.0, 742.0, 3937.0, 1237.0, 1091.0, 3214.0, 1273.0, 3809.0, 3115.0, 2111.0, 468.0, 3769.0, 2961.0, 3771.0, 246.0, 3094.0, 2907.0, 1016.0, 151.0, 377.0, 450.0, 3538.0, 3717.0, 2694.0, 2745.0, 2389.0, 3865.0, 281.0, 2272.0, 2991.0, 1810.0, 2024.0, 2725.0, 2731.0, 409.0, 2971.0, 1083.0, 2701.0, 1753.0, 1459.0, 2567.0, 673.0, 3516.0, 611.0, 947.0, 1176.0, 1640.0, 172.0, 2671.0, 2041.0, 2723.0, 2471.0, 378.0, 3901.0, 1834.0, 1733.0, 1135.0, 998.0, 2475.0, 292.0, 3347.0, 2121.0, 3952.0, 1219.0, 413.0, 2294.0, 1997.0, 849.0, 2017.0, 2025.0, 3476.0, 1399.0, 2822.0, 2068.0, 180.0, 2076.0, 3700.0, 1783.0, 3326.0, 1760.0, 2437.0, 3893.0, 2594.0, 16.0, 1942.0, 2171.0, 2815.0, 1281.0, 1589.0, 936.0, 3168.0, 2520.0, 3095.0, 3448.0, 1971.0, 1230.0, 3129.0, 3799.0, 3125.0, 3784.0, 3789.0, 3262.0, 1946.0, 2390.0, 1918.0, 3201.0, 3909.0, 2943.0, 2082.0, 3157.0, 2112.0, 3409.0, 1772.0, 1680.0, 3633.0, 2153.0, 720.0, 674.0, 3713.0, 126.0, 585.0, 2353.0, 158.0, 3676.0, 3398.0, 485.0, 765.0, 1284.0, 2089.0, 1148.0, 1147.0, 2183.0, 1037.0, 2393.0, 2250.0, 2524.0, 1617.0, 1457.0, 3135.0, 3142.0, 2935.0, 1461.0, 533.0, 1425.0, 1282.0, 728.0, 3521.0, 1972.0, 1361.0, 551.0, 2016.0, 454.0, 3889.0, 3837.0, 190.0, 2735.0, 2124.0, 2310.0, 23.0, 3548.0, 1466.0, 3743.0, 1124.0, 2033.0, 1590.0, 2138.0, 2716.0, 1649.0, 1189.0, 2135.0, 3243.0, 3359.0, 1339.0, 123.0, 1224.0, 2996.0, 344.0, 1101.0, 515.0, 2428.0, 1873.0, 1392.0, 2583.0, 258.0, 2519.0, 2771.0, 213.0, 451.0, 2906.0, 2313.0, 3253.0, 1343.0, 2941.0, 745.0, 2729.0, 353.0, 1707.0, 2859.0, 2108.0, 1359.0]
        L_f = [920.0, 3844.0, 2369.0, 1088.0, 3534.0, 1207.0, 17.0, 1041.0, 3512.0, 3418.0, 1188.0, 902.0, 2336.0, 3911.0, 1441.0, 141.0, 2690.0, 928.0, 39.0, 2762.0, 906.0, 838.0, 2657.0, 2125.0, 3565.0, 1967.0, 2291.0, 914.0, 932.0, 1620.0, 2160.0, 247.0, 222.0, 261.0, 2881.0, 2145.0, 3072.0, 1028.0, 1956.0, 2080.0, 1286.0, 3798.0, 1959.0, 28.0, 2248.0, 3247.0, 3594.0, 3155.0, 1345.0, 531.0, 1277.0, 593.0, 3044.0, 3083.0, 3005.0, 1380.0, 2020.0, 105.0, 1678.0, 1608.0, 2572.0, 3791.0, 1104.0, 2144.0, 318.0, 1186.0, 1073.0, 595.0, 2724.0, 1641.0, 351.0, 2908.0, 357.0, 3079.0, 1688.0, 3556.0, 3186.0, 2406.0, 224.0, 1962.0, 1480.0, 3251.0, 11.0, 345.0, 3526.0, 1784.0, 951.0, 3668.0, 2485.0, 1958.0, 2739.0, 916.0, 950.0, 2443.0, 3684.0, 904.0, 898.0, 587.0, 552.0, 2143.0, 3481.0, 3097.0, 3067.0, 1449.0, 47.0, 616.0, 3281.0, 1259.0, 661.0, 2348.0, 562.0, 3606.0, 2496.0, 2085.0, 1271.0, 372.0, 2857.0, 3325.0, 1394.0, 1081.0, 1032.0, 918.0, 1409.0, 314.0, 899.0, 733.0, 2245.0, 381.0, 2316.0, 232.0, 2405.0, 2677.0, 1066.0, 2396.0, 2282.0, 1059.0, 2622.0, 1941.0, 959.0, 3479.0, 3124.0, 1197.0, 1777.0, 915.0, 955.0, 1648.0, 3705.0, 3061.0, 34.0, 1285.0, 1.0, 2875.0, 1150.0, 3545.0, 2664.0, 2155.0, 1097.0, 262.0, 3915.0, 971.0, 2186.0, 3702.0, 3105.0, 2280.0, 3604.0, 3515.0, 1513.0, 2331.0, 1500.0, 2803.0, 945.0, 2639.0, 3051.0, 837.0, 3408.0, 457.0, 1801.0, 2506.0, 4.0, 2469.0, 270.0, 46.0, 1235.0, 2355.0, 2346.0, 1357.0, 461.0, 3255.0, 3176.0, 3350.0, 2975.0, 2014.0, 3936.0, 2072.0, 1353.0, 2006.0, 1397.0, 2612.0, 1099.0, 1367.0, 3270.0, 938.0, 2357.0, 94.0, 412.0, 1518.0, 3591.0, 538.0, 2000.0, 2846.0, 708.0, 329.0, 2995.0, 653.0, 1280.0, 5.0, 337.0, 1022.0, 2468.0, 1569.0, 905.0, 1031.0, 900.0, 1541.0, 2926.0, 3730.0, 1900.0, 2718.0, 1021.0, 3185.0, 2746.0, 327.0, 2805.0, 3101.0, 2920.0, 3269.0, 1674.0, 477.0, 3686.0, 2077.0, 2801.0, 581.0, 2133.0, 3724.0, 3296.0, 3554.0, 3478.0, 1479.0, 3720.0, 491.0, 1014.0, 1236.0, 3134.0, 695.0, 2763.0, 1013.0, 1096.0, 1856.0, 2827.0, 248.0, 1875.0, 3211.0, 3672.0, 215.0, 3224.0, 3396.0, 469.0, 1897.0, 3528.0, 2870.0, 917.0, 930.0, 1654.0, 3328.0, 3786.0, 907.0, 3870.0, 1422.0, 2206.0, 2114.0, 2324.0, 2575.0, 919.0, 3467.0, 1047.0, 1806.0, 350.0, 230.0, 2505.0, 48.0, 182.0, 144.0, 170.0, 2141.0, 1916.0, 3081.0, 1191.0, 1086.0, 2598.0, 546.0, 1407.0, 153.0, 2635.0, 2057.0, 2037.0, 1327.0, 3145.0, 446.0, 2193.0, 1337.0, 1913.0, 195.0, 2132.0, 1804.0, 3562.0, 3706.0, 1172.0, 1042.0, 2946.0, 2514.0, 1093.0, 1616.0, 3011.0, 2151.0, 1111.0, 613.0, 1043.0, 2774.0, 2154.0, 2621.0, 52.0, 3060.0, 3723.0, 206.0, 3133.0, 1821.0, 1964.0, 211.0, 2454.0, 532.0, 218.0, 3156.0, 1586.0, 1126.0, 2096.0, 927.0, 2007.0, 778.0, 2097.0, 3117.0, 691.0, 3567.0, 1223.0, 1268.0, 1300.0, 2747.0, 1573.0, 3302.0, 671.0, 3471.0, 3825.0, 1064.0, 1299.0, 252.0, 3004.0, 2091.0, 2337.0, 61.0, 1020.0, 3763.0, 1727.0, 74.0, 3599.0, 3708.0, 465.0, 29.0, 3741.0, 3457.0, 2399.0, 781.0, 69.0, 3635.0, 3808.0, 3249.0, 2732.0, 1621.0, 1686.0, 3435.0, 3857.0, 3299.0, 3426.0, 176.0, 343.0, 2972.0, 2853.0, 272.0, 2788.0, 1393.0, 203.0, 1465.0, 801.0, 1917.0, 2431.0, 3714.0, 2967.0, 3553.0, 79.0, 3951.0, 1683.0, 3071.0, 3102.0, 302.0, 3655.0, 2261.0, 3877.0, 2266.0, 3716.0, 3699.0, 1769.0, 266.0, 1173.0, 2693.0, 3093.0, 1658.0, 277.0, 279.0, 848.0, 839.0, 2365.0, 2738.0, 1264.0, 271.0, 1269.0, 2043.0, 3855.0, 1030.0, 1346.0, 2052.0, 2142.0, 2719.0, 2574.0, 2053.0, 1410.0, 3912.0, 1381.0, 3660.0, 2446.0, 2613.0, 2314.0, 978.0, 348.0, 2168.0, 3466.0, 669.0, 3649.0, 2448.0, 2899.0, 1611.0, 2940.0, 8.0, 1463.0, 26.0, 3557.0, 1994.0, 1758.0, 414.0, 1027.0, 3088.0, 3391.0, 1936.0, 2205.0, 3861.0, 332.0, 3450.0, 2585.0, 3618.0, 425.0, 1605.0, 3827.0, 846.0, 2267.0, 2359.0, 2952.0, 2786.0, 3923.0, 1290.0, 3240.0, 3388.0, 1547.0, 338.0, 3712.0, 3063.0, 242.0, 715.0, 3679.0, 3571.0, 668.0, 1069.0, 2276.0, 1438.0, 2688.0, 2900.0, 168.0, 3539.0, 199.0, 3675.0, 2436.0, 647.0, 724.0, 82.0, 542.0, 1362.0, 117.0, 2109.0, 3246.0, 3019.0, 1904.0, 360.0, 2966.0, 482.0, 2741.0, 334.0, 2100.0, 2173.0, 1615.0, 358.0, 280.0, 3932.0, 369.0, 3547.0, 3739.0, 1788.0, 875.0, 2106.0, 3719.0, 3839.0, 1417.0, 3566.0, 3795.0, 670.0, 520.0, 208.0, 3449.0, 3274.0, 27.0, 3872.0, 2969.0, 2927.0, 2442.0, 113.0, 2084.0, 1848.0, 3882.0, 3790.0, 3926.0, 2820.0, 3922.0, 3046.0, 832.0, 3896.0, 2101.0, 1600.0, 2548.0, 2453.0, 386.0, 239.0, 1015.0, 85.0, 3077.0, 3264.0, 3340.0, 3114.0, 1729.0, 2498.0, 309.0, 1034.0, 2421.0, 3438.0, 2599.0, 405.0, 3461.0, 3813.0, 3238.0, 3399.0, 3921.0, 912.0, 1840.0, 2876.0, 319.0, 40.0, 257.0, 3287.0, 880.0, 754.0, 1874.0, 2241.0, 2553.0, 1699.0, 550.0, 1549.0, 2338.0, 1922.0, 3612.0, 1894.0, 1049.0, 1185.0, 2779.0, 3902.0, 3580.0, 2.0, 2435.0, 73.0, 1012.0, 1275.0, 783.0, 512.0, 1919.0, 3838.0, 2903.0, 507.0, 1896.0, 2263.0, 2320.0, 1515.0, 363.0, 3492.0, 1562.0, 1588.0, 408.0, 3405.0, 307.0, 1199.0, 3268.0, 186.0, 1961.0, 1428.0, 2540.0, 3284.0, 2062.0, 3624.0, 1169.0, 2513.0, 575.0, 380.0, 2696.0, 2070.0, 2130.0, 3897.0, 615.0, 50.0, 3852.0, 415.0, 1797.0, 1660.0, 506.0, 3704.0, 2816.0, 2678.0, 2122.0, 1836.0, 2126.0, 481.0, 87.0, 3577.0, 2990.0, 3200.0, 441.0, 1554.0, 346.0, 1653.0, 2202.0, 2616.0, 283.0, 3584.0, 2417.0, 2284.0, 2042.0, 3454.0, 1582.0, 2568.0, 1669.0, 2048.0, 3613.0, 1911.0, 949.0, 420.0, 1719.0, 2361.0, 41.0, 3949.0, 379.0, 2379.0, 3447.0, 2136.0, 2642.0, 3206.0, 1995.0, 3150.0, 2856.0, 2010.0, 2532.0, 382.0, 2398.0, 1798.0, 1242.0, 2414.0, 2550.0, 1084.0, 131.0, 3055.0, 2630.0, 1949.0, 1954.0, 2352.0, 2110.0, 3181.0, 2021.0, 1344.0, 3685.0, 1398.0, 1312.0, 910.0, 3738.0, 173.0, 1456.0, 3445.0, 986.0, 2848.0, 2722.0, 3696.0, 3864.0, 3707.0, 1171.0, 558.0, 356.0, 2717.0, 3204.0, 2561.0, 934.0, 2704.0, 371.0, 1831.0, 879.0, 2439.0, 3108.0, 2517.0, 1372.0, 1672.0, 807.0, 3616.0, 688.0, 2797.0, 519.0, 1211.0, 1730.0, 1446.0, 1546.0, 2445.0, 2147.0, 3475.0, 1556.0, 1580.0, 1220.0, 2373.0, 501.0, 124.0, 1216.0, 1429.0, 2683.0, 2066.0, 1881.0, 2949.0, 3090.0, 802.0, 1870.0, 407.0, 586.0, 1944.0, 2989.0, 1921.0, 1226.0, 2380.0, 3489.0, 3886.0, 2190.0, 2919.0, 2495.0, 2392.0, 753.0, 1484.0, 1667.0, 2363.0, 3308.0, 1077.0, 1805.0, 2714.0, 3173.0, 216.0, 1694.0, 736.0, 1321.0, 1483.0, 608.0, 1485.0, 1347.0, 2789.0, 25.0, 2699.0, 1792.0, 2065.0, 2709.0, 2860.0, 1845.0, 2752.0, 494.0, 2273.0, 62.0, 2710.0, 866.0, 3841.0, 1566.0, 3153.0, 973.0, 3600.0, 1240.0, 1270.0, 923.0, 2159.0, 896.0, 3258.0, 147.0, 3439.0, 2947.0, 2643.0, 1212.0, 1258.0, 2527.0, 1419.0, 1217.0, 316.0, 1293.0, 2420.0, 3130.0, 2474.0, 2879.0, 991.0, 3317.0, 2713.0, 3440.0, 2463.0, 1619.0, 2539.0, 3070.0, 3040.0, 2163.0, 508.0, 428.0, 1816.0, 2533.0, 2736.0, 1969.0, 3054.0, 2176.0, 288.0, 2794.0, 2239.0, 2290.0, 1234.0, 3735.0, 2166.0, 19.0, 2071.0, 2394.0, 2858.0]

    elif dataset == 'yahoo':
        X = DL.load_user_item_matrix_yahoo()
        T = DL.load_gender_vector_yahoo()
        X_filled = DL.load_user_item_matrix_yahoo_Impute()
        # updated list with each users min rating 20
        L_m = [2587, 3581, 4289, 4049, 132, 916, 7038, 1111, 6790, 1691, 372, 5818, 7266, 1946, 3713, 7661, 2450, 6177, 1487, 4249, 6787, 6262, 4743, 6590, 7262, 8346, 7565, 5073, 5061, 5003, 1442, 7660, 1409, 7064, 2956, 7451, 3425, 1367, 5300, 5908, 7063, 2858, 3210, 292, 7288, 6750, 3123, 4507, 1278, 5373, 5040, 1134, 7895, 6763, 6539, 1483, 2802, 2998, 1066, 4016, 6547, 5164, 3471, 1430, 5532, 1556, 1106, 3239, 3887, 4217, 1415, 7558, 3582, 3534, 6574, 4343, 5729, 762, 6635, 4639, 802, 8568, 3948, 3724, 5577, 4789, 3326, 4481, 6185, 1165, 6811, 5592, 1615, 3755, 6376, 2590, 3258, 6582, 5582, 1376, 1799, 3199, 1555, 5227, 4358, 5265, 4522, 144, 6858, 8287, 1863, 6925, 6292, 6412, 6482, 4004, 5216, 7220, 7759, 2686, 2925, 5130, 2368, 177, 2366, 5013, 3249, 3245, 5937, 578, 2260, 984, 1351, 8141, 3940, 5555, 2115, 4459, 8315, 2693, 1867, 4252, 8136, 3153, 3186, 4056, 3487, 1947, 5935, 769, 1744, 6789, 5814, 4962, 6116, 2677, 8529, 4870, 3570, 6718, 4068, 2947, 1805, 5043, 6455, 6992, 6067, 2930, 3394, 6270, 4244, 7601, 8464, 2648, 5796, 6165, 2815, 5972, 6753, 6857, 3317, 3630, 327]
        L_f = [8569, 8176, 8494, 5099, 8218, 8533, 4931, 126, 760, 7813, 8563, 4468, 8219, 562, 8319, 4636, 1100, 8215, 8379, 1642, 8072, 8323, 3618, 7020, 7864, 7628, 4804, 441, 323, 719, 5302, 7885, 8390, 2315, 8306, 8238, 8301, 8253, 1160, 2405, 1970, 8177, 6944, 5675, 8093, 7656, 1576, 1362, 550, 4819, 6957, 939, 4234, 2258, 6970, 5448, 352, 7651, 7490, 8349, 7600, 54, 7781, 6221, 100, 7478, 92, 8430, 7081, 7587, 5039, 4233, 7592, 2972, 7498, 8506, 4903, 7778, 282, 8235, 6801, 6357, 8474, 303, 7972, 7630, 1621, 6948, 5984, 5391, 52, 36, 6991, 4464, 4893, 7883, 8039, 423, 7732, 3964, 291, 8531, 235, 5225, 1971, 1292, 4280, 5291, 7589, 210, 654, 361, 7557, 8459, 7834, 8134, 6932, 8227, 1101, 587, 7983, 4274, 606, 6967, 7005, 634, 7590, 110, 5841, 7860, 5521, 215, 4010, 542, 7996, 7466, 7990, 7644, 8418, 616, 8425, 8470, 8033, 388, 7756, 382, 5967, 7769, 4486, 5464, 7768, 384, 7705, 6761, 8370, 1908, 8092, 8318, 8398, 5825, 6937, 7772, 8362, 7703, 485, 7835, 8123, 5443, 2023, 8165, 5623, 7737, 5890, 8249, 2906, 4629, 8188, 149, 468, 407, 7987, 4892, 8003, 7964, 5376, 5687, 7655, 563, 6910, 4963, 7999, 7796, 8041, 4741, 4203, 4699, 8485, 6895, 5529, 6193, 7896, 597, 5159, 8027, 7479, 818, 7798, 6587, 601, 3807, 3, 8572, 6904, 2052, 621, 8266, 5850, 7483, 8048, 3941, 8486, 5404, 7936, 5134, 8303, 325, 3831, 8057, 6405, 8157, 373, 3013, 5621, 87, 6894, 8071, 5614, 5605, 8091, 8274, 8206, 329, 6488, 6837, 3826, 2323, 5025, 2494, 8058, 62, 3071, 6174, 5884, 5838, 7707, 5865, 8070, 356, 1250, 5539, 125, 7240, 7949, 3859, 182, 69, 4271, 8481, 8061, 5630, 7854, 3509, 7958, 10, 7502, 4525, 8083, 6462, 7873, 557, 154, 137, 5956, 7809, 8180, 6105, 3357, 5307, 5485, 1285, 6343, 7612, 8046, 7459, 6922, 70, 1713, 855, 8438, 204, 7945, 2085, 4337, 7747, 7633, 7941, 7012, 7843, 2567, 2522, 1872, 620, 558, 8226, 4851, 7973, 8233, 527, 7697, 213, 8214, 8421, 541, 8240, 4466, 6756, 7912, 8060, 5421, 5665, 2155, 3881, 7994, 3195, 7131, 8205, 2555, 6930, 7539, 4152, 7731, 3697, 7968, 607, 271, 5809, 239, 11, 4424, 3692, 8232, 7, 6239, 8295, 3073, 365, 5974, 1922, 1136, 2986, 7853, 2227, 7795, 6839, 1213, 8239, 524, 8124, 5579, 5530, 276, 519, 4813, 5552, 5034, 7704, 7788, 8162, 5308, 2234, 1416, 8472, 8225, 3292, 7842, 8154, 6999, 5342, 4514, 7634, 7613, 820, 8095, 3794, 6161, 102, 8327, 253, 7810, 7443, 1043, 107, 3949, 7741, 25, 254, 4981, 7519, 411, 3838, 7829, 8155, 8208, 5768, 8312, 3293, 6015, 7525, 8203, 71, 6997, 8244, 8326, 5755, 7579, 8343, 2017, 2386, 6849, 5052, 6091, 2583, 2088, 422, 8490, 4769, 469, 514, 6897, 575, 763, 3314, 3883, 8160, 8189, 5672, 7211, 5553, 6266, 8211, 7706, 8350, 805, 455, 5058, 4900, 403, 7775, 7876, 7461, 217, 6908, 8037, 7620, 7982, 1785, 7955, 8434, 8289, 7098, 6018, 5919, 5878, 5245, 8537, 8273, 538, 7838, 3614, 8212, 8193, 8210, 13, 8030, 5800, 842, 8250, 8269, 3580, 6211, 7520, 566, 8143, 8373, 6444, 3337, 8159, 4258, 7708, 8204, 7848, 7567, 151, 7937, 694, 473, 8190, 248, 8264]

    avg_ratings = np.zeros(shape=X.shape[1]) # --- will store item avg. rating
    initial_count = np.zeros(shape=X.shape[1])

    for item_id in range(X.shape[1]):
        ratings = [rating for rating in X[:, item_id] if rating > 0 ]
        avg_ratings[item_id] = np.average(ratings) if ratings else 0
        initial_count[item_id] = len(ratings)

    max_count = initial_count * notice_factor
    # max_count: that means how much a item can be added to the users profile is not considered in blurme

    print("obfuscation")

    # Now, start obfuscating the data:

    X_obf = np.copy(X)

    prob_m = []#[p / sum(C_m) for p in C_m]
    prob_f = []#[p / sum(C_f) for p in C_f]

    for index, user in enumerate(X):
        rate = sum(1 for rating in user if rating > 0)
        k = rate * p
        print(f"User: {index}, no. rate: {rate} and k = {k}, sex: {T[index]}")
        greedy_index = 0

        if T[index] == 1: # female user
            added = 0
            safety_counter = 0

            # select a movie for obfuscation
            while added < k and safety_counter < 100:
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                elif sample_mode == 'sampled':
                    movie_id = L_m[np.random.choice(range(len(L_m)), p=prob_m)]
                elif sample_mode == 'greedy':
                    movie_id = L_m[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_m):
                        safety_counter = 100
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue

                # set rating of the selected movie
                if X_obf[index, int(movie_id)-1] == 0:# and X_test [index, int(movie_id) - 1] ==0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int((movie_id) - 1) ] # avg_ratings[int(index)]
                    elif rating_mode == 'pred':
                        X_obf[index, int(movie_id) - 1] = X_filled[index, int(movie_id) - 1]

                    added += 1
                safety_counter += 1
            print(f"user: {index}, item added: {added}, movie: {movie_id}, rating: {X_obf[index, int(movie_id) - 1]}")


        elif T[index] == 0: # male user
            added = 0
            safety_counter = 0
            while added < k and safety_counter < 100:
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                elif sample_mode == 'sampled':
                    movie_id = L_f[np.random.choice(range(len(L_f)), p=prob_f)]
                elif sample_mode == 'greedy':
                    movie_id = L_f[greedy_index]
                    greedy_index += 1
                    if greedy_index >= len(L_f):
                        safety_counter = 100

                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue

                # set the rating of the selected item
                if X_obf[index, int(movie_id) - 1] == 0:# and X_test [index, int(movie_id) - 1] ==0:
                    if rating_mode == 'higest':
                        X_obf[index, int(movie_id) - 1] = 5
                    elif rating_mode == 'avg':
                        X_obf[index, int(movie_id) - 1] = avg_ratings[int((movie_id) - 1) ] #int(index)
                    elif rating_mode == 'pred':
                        X_obf[index, int(movie_id) - 1] = X_filled[index, int(movie_id) - 1]
                    added += 1
                safety_counter += 1

                print(f"user: {index}, item added: {added}, movie: {movie_id}, rating: {X_obf[index, int(movie_id) - 1]}")

    # Save the obfuscated data to a file
    output_file = 'ml-' + dataset + '/BlurMe/'
    print(output_file)
    with open(output_file + "Blur_" + rating_mode + "_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(int(np.round(rating))) + "::000000000\n")

    return X_obf


def blurMePP():
    top = -1
    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    removal_mode = list(['random', 'strategic'])[1]
    rating_mode = list(['avg', 'predicted'])[0]
    #id_index, index_id = MD.load_movie_id_index_dict()
    notice_factor = 2
    p = 0.05
    dataset = ['ML', 'Fx', 'LFM', 'Li'][0]
    if dataset == 'ML':
        X = MD.load_user_item_matrix_1m_all()
        # X = MD.load_user_item_matrix_1m_trainingSet()  # load_user_item_matrix_1m_trainingSet max_user=max_user, max_item=max_item)
        T = MD.load_gender_vector_1m()  # max_user=max_user)
        #X_test = MD.load_user_item_matrix_1m_testSet()
        # X = MD.load_user_item_matrix_100k()
        # T = MD.load_gender_vector_100k()
    elif dataset == 'Fx':
        """import FlixsterData as FD
        #X, T, _ = FD.load_flixster_data_subset()
        X, T, _ = FD.load_flixster_data_subset_trainingSet()"""
        import FlixsterDataSub as FDS
        # X = FDS.load_user_item_matrix_FX_All()
        X = FDS.load_user_item_matrix_FX_TrainingSet()
        X_test = FDS.load_user_item_matrix_FX_Test()
        T = FDS.load_gender_vector_FX()
    elif dataset == 'LFM':
        print("no file for lfm")
        #import LastFMData as LFM
        # X = LFM.load_user_item_matrix_lfm_Train()  # LFM.load_user_item_matrix_lfm_All()
        #X = LFM.load_user_item_matrix_lfm_All()  # load_user_item_matrix_lfm_Train LFM.load_user_item_matrix_lfm_All()
        #T = LFM.load_gender_vector_lfm()
        #X_test = LFM.load_user_item_matrix_lfm_Test()
    else:
        import LibimSeTiData as LD
        X, T, _ = LD.load_libimseti_data_subset()
    # X = Utils.normalizze(X)
    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])

    for item_id in range(X.shape[1]):
        ratings = [rating for rating in X[:, item_id] if rating > 0]
        avg_ratings[item_id] = np.average(ratings) if ratings else 0
        initial_count[item_id] = len(ratings)

    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    """from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    cv = StratifiedKFold(n_splits=10)
    coefs = []
    avg_coefs = np.zeros(shape=(len(X[1]),))

    random_state = np.random.RandomState(0)
    for train, test in cv.split(X, T):
        x, t = X[train], T[train]
        model = LogisticRegression(penalty='l2', random_state=random_state)
        model.fit(x, t)
        # rank the coefs:
        ranks = ss.rankdata(model.coef_[0])
        coefs.append(ranks)
        # print(len(model.coef_[0]),len(x[0]))
        avg_coefs += model.coef_[0]

    coefs = np.average(coefs, axis=0)
    coefs = [[coefs[i], i + 1, avg_coefs[i]] for i in range(len(coefs))]
    coefs = np.asarray(list(sorted(coefs)))


    if top == -1:
        values = coefs[:,2]
        index_zero = np.where(np.abs(values) == np.min(np.abs(values)))
        top_male = index_zero[0][0]
        top_female = index_zero[0][-1]
        L_m = coefs[:top_male, 1][100:]
        # print(len(L_m))
        R_m = 2835 - coefs[:top_male, 0] #3952 2835
        C_m = np.abs(coefs[:top_male, 2])
        # C_m = [x for x in C_m if x > 2] # C_m[C_m <=  2]
        # print("C_m", type (C_m), "\n", C_m)
        L_f = coefs[coefs.shape[0] - top_female:, 1][100:]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0] - top_female:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0] - top_female:, 2]
        C_f = list(reversed(np.abs(C_f)))
        # C_f = [x for x in C_f if x > 2] # C_f[C_f <= 2]
        # print("C_f", type (C_f), "\n", C_f)

        # plt.plot(C_m, label = 'Male Coef', c= 'lightskyblue')
        # plt.plot(C_f, label = 'Female Coef', c= 'lightpink')
        # plt.axhline(y=2, color='crimson', linestyle='--')
        # plt.legend(loc="upper right")
        # plt.title("Male and Female coefficients on Flixster Data", fontsize=16, fontweight="bold")
        # plt.xlabel ('Features', fontsize=19)
        # plt.ylabel ('Coefficients', fontsize=19)
        # # plt.savefig("threshold_ML1M_IndicativeItems.pdf")
        # plt.show()

    else:
        L_m = coefs[:top, 1]
        R_m = 2835 -coefs[:top, 0] #3952 2835
        C_m = np.abs(coefs[:top, 2])
        L_f = coefs[coefs.shape[0]-top:, 1]
        L_f = list(reversed(L_f))
        R_f = coefs[coefs.shape[0]-top:, 0]
        R_f = list(reversed(R_f))
        C_f = coefs[coefs.shape[0]-top:, 2]
        C_f = list(reversed(np.abs(C_f)))

    # print(len(L_f))
    # Here we are trying to get all the less indicative items for F / M
    # Based on the plot we see that from 600 to the end the coefficients are <= 2
    L_ff = L_f.copy()
    print(L_ff)
    ## low indicative items
    # L_ff = L_ff [100:]
    # highly indicative items
    # L_ff = L_ff[:400]
    # print("L_ff:", L_ff, "\n\n", len(L_ff))
    L_ff = pd.DataFrame(L_ff)
    L_ff.to_csv('L_f_FX_Normalized.csv', index=False)
    # print("------")
    # print(len(L_m))
    L_mm = L_m.copy()
    print (L_mm)
    ## low indicative items
    # L_mm = L_mm [100:]
    # highly indicative items
    # L_mm = L_mm[:400]
    # print("L_mm:", L_mm, "\n\n", len( L_mm))
    L_mm = pd.DataFrame(L_mm)
    L_mm.to_csv('L_m_FX_Normalized.csv', index=False)"""
    # Now, where we have the two lists, we can start obfuscating the data:
    #X = MD.load_user_item_matrix_1m()
    #np.random.shuffle(X)
    #print(X.shape)

    X_obf = np.copy(X)
    total_added = 0
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index_m = 0
        greedy_index_f = 0
        # print(k)
        added = 0
        if T[index] == 1:
            safety_counter = 0
            while added < k and safety_counter < 100:
                if greedy_index_m >= len(L_m):
                    safety_counter = 100
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_m[greedy_index_m]
                if sample_mode == 'random':
                    movie_id = L_m[np.random.randint(0, len(L_m))]
                greedy_index_m += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id)-1]])
                if rating_count > max_count[int(movie_id)-1]:
                    continue
                if X_obf[index, int(movie_id) - 1] == 0:# and X_test [index, int(movie_id) - 1] ==0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        elif T[index] == 0:
            safety_counter = 0
            while added < k and safety_counter < 100:
                if greedy_index_f >= len(L_f):
                    safety_counter = 100
                    continue
                if sample_mode == 'greedy':
                    movie_id = L_f[greedy_index_f]
                if sample_mode == 'random':
                    movie_id = L_f[np.random.randint(0, len(L_f))]
                greedy_index_f += 1
                rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, int(movie_id) - 1]])
                if rating_count > max_count[int(movie_id) - 1]:
                    continue

                if X_obf[index, int(movie_id) - 1] == 0:# and X_test [index, int(movie_id) - 1] ==0:
                    X_obf[index, int(movie_id) - 1] = avg_ratings[int(movie_id) - 1]
                    added += 1
                safety_counter += 1
        total_added += added

    # Now remove ratings from users that have more than 200 ratings equally:
    if removal_mode == "random":
        nr_many_ratings = 0
        for user in X:
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20: # 200 for ML1M and 300 for Flixster
                nr_many_ratings += 1
        nr_remove = total_added / nr_many_ratings

        for user_index, user in enumerate(X):
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0], size=(int(nr_remove),))#,replace=False)
                X_obf[user_index, to_be_removed_indecies] = 0
    else:
        nr_many_ratings = 0
        for user in X:
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                nr_many_ratings += 1
        print("nr_many_ratings:", nr_many_ratings)
        print("total_added:", total_added)
        nr_remove = total_added / nr_many_ratings

        for user_index, user in enumerate(X):
            print("user: ", user_index)
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                index_m = 0
                index_f = 0
                rem = 0
                if T[user_index] == 1:
                    safety_counter = 0
                    # We note that if we add safety_counter < 1000 in the while we have a higher accuracy than if we keep it in the if
                    while (rem < nr_remove) and safety_counter < 100:
                        if index_f >= len(L_f) :
                            safety_counter = 100
                            continue

                        if removal_mode == "random":
                            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                      size=(int(nr_remove),),
                                                                      replace=False)  # , replace=False)
                        if removal_mode == "strategic":
                            to_be_removed_indecies = L_f[index_f]
                        index_f += 1

                        if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                            X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                            rem += 1
                        safety_counter += 1

                elif T[user_index] == 0:

                    while (rem < nr_remove) and safety_counter < 100:
                        if index_m >= len(L_m) :#and safety_counter < 1000:
                            safety_counter = 100
                            continue

                        if removal_mode == "random":
                            to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0],
                                                                      size=(int(nr_remove),),
                                                                      replace=False)  # , replace=False)
                        # X_obf[user_index, to_be_removed_indecies] = 0

                        if removal_mode == "strategic":
                            to_be_removed_indecies = L_m[index_m]
                        index_m += 1

                        if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                            X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                            rem += 1
                        safety_counter += 1


    # output the data in a file:
    output_file = ""
    if dataset == 'ML':
        output_file = "ml-1m/BlurMore/"#"ml1m/"#"ml-1m/BlurMore/" ml-1m/BlurMore/Random_Removal/
        with open(output_file + "All_testSafe`Count_threshold20_ML1M_blurmepp_obfuscated_" + sample_mode + "_" +
                  str(p) + "_" + str(notice_factor) + "_" + str(removal_mode)  + ".dat",
                  'w') as f: # + "_" + str(removal_mode) + ".dat",
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(
                            str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                                int(np.round(rating))) + "::000000000\n")

    elif dataset == 'Fx':
        output_file = "Flixster/"#BlurMore/RandomRem/" # "Flixster/BlurMore/Greedy_Removal/" FX/
        with open(output_file + "All_testSafe`Count_threshold20_ExcludeTestSet_FX_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(
                notice_factor) + "_" + str(removal_mode) + ".dat",
                  'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")
    elif dataset == 'LFM':
        output_file = "lastFM/"#BlurMore/RandomRem/"
        with open(output_file + "All_testSafe`Count_LFM_blurmepp_ExcludeTestSet_obfuscated_" + sample_mode + "_" + str(p) + "_" +str( notice_factor) + "_" + str(removal_mode) + ".dat",
                      'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")

    else:
        with open("libimseti/LST_blurmepp_obfuscated_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat", 'w') as f:
            for index_user, user in enumerate(X_obf):
                for index_movie, rating in enumerate(user):
                    if rating > 0:
                        f.write(str(index_user+1) + "::" + str(index_movie+1) + "::" + str(
                            int(np.round(rating))) + "::000000000\n")


    return X_obf


# --- Creation of Personalized list of indicative items

def Personalized_list_User(dataset):
    notice_factor = 2
    item_choice = {}

    ### this is actually neighbor list for each missing item
    with open('ml-' +dataset+'/NN_All_AllUsers_Neighbors_Weight_K_30_item_choice.json') as json_file:
        data = json.load(json_file)
    len_dict = {}

    print("data loaded")
    for key, value in data.items():
        length = []
        for v in value:
            length.append(len(v))
        len_dict[int(key)] = length

    if dataset == '100k':
        X = DL.load_user_item_matrix_100k()
        T = DL.load_gender_vector_100k()
        L_m = [751, 327, 24, 688, 472, 888, 317, 948, 315, 178, 305, 179, 359, 186, 10, 883, 294, 342, 5, 471, 1048, 1101, 879, 449, 324, 150, 357, 411, 1265, 136, 524, 678, 202, 692, 100, 303, 264, 339, 1010, 276, 218, 995, 240, 433, 271, 203, 544, 520, 410, 222, 1028, 924, 823, 322, 462, 265, 750, 23, 281, 137, 333, 736, 164, 358, 301, 479, 647, 168, 144, 94, 687, 60, 70, 175, 6, 354, 344, 1115, 597, 596, 42, 14, 521, 184, 393, 188, 450, 1047, 682, 7, 760, 737, 270, 1008, 260, 295, 1051, 289, 430, 235, 258, 330, 56, 347, 52, 205, 371, 1024, 8, 369, 116, 79, 831, 694, 287, 435, 495, 242, 33, 201, 230, 506, 979, 620, 355, 181, 717, 936, 500, 893, 489, 1316, 855, 2, 180, 871, 755, 523, 477, 227, 412, 183, 657, 510, 77, 474, 1105, 67, 705, 362, 1133, 511, 546, 768, 96, 457, 108, 1137, 352, 341, 825, 165, 584, 505, 908, 404, 679, 15, 513, 117, 490, 665, 28, 338, 59, 32, 1014, 989, 351, 475, 57, 864, 969, 177, 316, 463, 134, 703, 306, 378, 105, 99, 229, 484, 651, 157, 232, 114, 161, 395, 754, 931, 591, 408, 685, 97, 554, 468, 455, 640, 473, 683, 300, 31, 1060, 650, 72, 191, 259, 1280, 199, 826, 747, 68, 1079, 887, 578, 329, 274, 174, 428, 905, 780, 753, 396, 4, 277, 71, 519, 1062, 189, 325, 502, 233, 1022, 880, 1063, 197, 813, 16, 331, 208, 162, 11, 963, 501, 820, 930, 896, 318, 1142, 1194, 425, 171, 282, 496, 26, 215, 573, 527, 730, 49, 693, 517, 336, 417, 207, 900, 299, 226, 606, 268, 192, 407, 62, 636, 480, 1016, 241, 945, 343, 603, 231, 469, 515, 492, 664, 972, 642, 781, 984, 187, 149, 257, 40, 141, 978, 44, 845, 85, 244, 512, 182, 1021, 756, 1, 778, 644, 1050, 185, 530, 81, 497, 3, 335, 923, 509, 418, 724, 566, 221, 570, 655, 413, 1311, 340, 583, 537, 635, 686, 176, 296, 926, 561, 101, 173, 862, 680, 652]
        L_f = [332, 321, 877, 906, 827, 882, 337, 278, 872, 304, 87, 93, 1038, 292, 629, 727, 220, 898, 107, 143, 516, 819, 382, 499, 876, 45, 1600, 111, 416, 132, 752, 125, 689, 604, 628, 310, 246, 419, 148, 659, 881, 248, 899, 707, 20, 129, 272, 690, 350, 498, 19, 216, 713, 429, 253, 372, 311, 251, 288, 1023, 392, 1328, 131, 320, 83, 895, 912, 445, 1094, 432, 109, 156, 532, 588, 486, 653, 243, 319, 518, 269, 298, 447, 290, 1386, 988, 423, 723, 279, 696, 568, 919, 1296, 266, 645, 937, 595, 482, 748, 348, 749, 815, 451, 154, 684, 993, 904, 204, 198, 403, 525, 553, 671, 507, 1119, 1036, 65, 420, 169, 9, 427, 832, 894, 729, 654, 1128, 987, 1281, 401, 1089, 942, 190, 406, 170, 402, 878, 291, 710, 297, 721, 151, 367, 954, 856, 64, 312, 385, 581, 370, 39, 648, 691, 746, 89, 1033, 745, 194, 662, 145, 539, 98, 147, 739, 159, 90, 236, 55, 120, 448, 200, 409, 623, 118, 731, 365, 238, 124, 146, 716, 1026, 847, 261, 153, 262, 130, 127, 866, 58, 356, 195, 1086, 1190, 1041, 1073, 889, 405, 950, 891, 611, 582, 66, 1035, 346, 1394, 172, 909, 1013, 785, 925, 488, 704, 1315, 708, 1431, 309, 121, 239, 902, 213, 446, 1012, 1054, 873, 193, 478, 829, 916, 353, 349, 133, 249, 550, 638, 452, 1313, 466, 529, 458, 792, 313, 625, 668, 1061, 742, 1065, 763, 73, 846, 1090, 863, 1620, 890, 1176, 1017, 237, 1294, 245, 508, 387, 53, 514, 422, 1068, 1527, 939, 1232, 1011, 631, 381, 956, 875, 459, 607, 1442, 155, 697, 1066, 1285, 293, 88, 1221, 1109, 675, 254, 209, 649, 140, 48, 613, 962, 1157, 991, 1083, 720, 535, 901, 436, 286, 559, 1085, 892, 102, 211, 285, 1383, 758, 1234, 674, 1163, 283, 637, 735, 885, 630, 709, 938, 38, 142, 219, 953, 275, 1059, 676, 210, 47, 63, 255, 494, 744, 250, 1114, 672, 308, 632, 17, 1203, 762, 1522, 538, 491, 307, 669, 775, 1135, 217, 841, 949, 766, 966, 314, 812, 821, 574, 1039, 1388, 51, 579, 252, 50, 1009, 1040, 1147, 224, 212, 1483, 1278, 122, 934, 702, 443, 86, 1444, 69, 400, 975, 35, 549, 928, 531, 622, 614, 1020, 658, 740, 1468, 22, 1007, 470, 1269, 633, 1074, 1299, 1095, 1148, 280, 1136, 92, 167, 434, 284, 961, 1084, 126, 619, 974, 196, 485, 1152, 673, 627, 345, 29, 605, 929, 952, 714, 1053, 526, 123, 476, 660, 955, 380, 1503, 163, 493, 1197, 431, 504, 886, 1037, 13, 794, 273, 844, 1032, 1025, 106, 91, 533, 421, 699, 869, 78, 1243, 481, 661, 82, 732]

    elif dataset == '1m':
        X = DL.load_user_item_matrix_1m()
        T = DL.load_gender_vector_1m()
        L_m = [589.0, 1233.0, 2706.0, 1090.0, 2959.0, 1250.0, 2427.0, 2490.0, 1208.0, 1266.0, 3654.0, 1748.0, 1262.0, 1396.0, 1374.0, 2700.0, 1265.0, 1089.0, 1222.0, 231.0, 2770.0, 1676.0, 2890.0, 1228.0, 1136.0, 3360.0, 3298.0, 1663.0, 3811.0, 2011.0, 1261.0, 233.0, 3361.0, 2366.0, 1127.0, 1276.0, 3555.0, 1214.0, 3929.0, 299.0, 1304.0, 3468.0, 1095.0, 150.0, 1213.0, 750.0, 3082.0, 6.0, 111.0, 3745.0, 349.0, 541.0, 2791.0, 785.0, 1060.0, 1294.0, 1302.0, 1256.0, 1292.0, 2948.0, 3683.0, 3030.0, 3836.0, 913.0, 2150.0, 32.0, 2826.0, 2721.0, 590.0, 3623.0, 2997.0, 3868.0, 3147.0, 1610.0, 3508.0, 2046.0, 21.0, 1249.0, 10.0, 1283.0, 3760.0, 2712.0, 3617.0, 3552.0, 3256.0, 1079.0, 3053.0, 1517.0, 2662.0, 1953.0, 2670.0, 3578.0, 2371.0, 3334.0, 2502.0, 2278.0, 364.0, 3462.0, 2401.0, 3163.0, 2311.0, 852.0, 2916.0, 1378.0, 3384.0, 524.0, 70.0, 370.0, 3035.0, 3513.0, 2917.0, 3697.0, 24.0, 1957.0, 3494.0, 1912.0, 3752.0, 2013.0, 3452.0, 3928.0, 2987.0, 431.0, 2759.0, 1387.0, 1882.0, 3638.0, 1288.0, 2867.0, 2728.0, 2433.0, 161.0, 3386.0, 517.0, 741.0, 1287.0, 1231.0, 3062.0, 2288.0, 3753.0, 529.0, 3793.0, 3052.0, 2447.0, 1320.0, 3819.0, 1303.0, 922.0, 3022.0, 260.0, 858.0, 493.0, 3006.0, 480.0, 2410.0, 333.0, 1178.0, 3814.0, 2702.0, 1203.0, 2922.0, 1625.0, 3366.0, 3213.0, 2188.0, 2628.0, 3358.0, 2648.0, 3788.0, 953.0, 999.0, 3754.0, 3910.0, 3016.0, 3863.0, 303.0, 3263.0, 1080.0, 786.0, 3764.0, 2105.0, 3543.0, 2607.0, 3681.0, 592.0, 145.0, 2303.0, 1682.0, 1019.0, 3646.0, 1544.0, 235.0, 908.0, 3615.0, 2792.0, 2167.0, 2455.0, 1587.0, 1227.0, 2901.0, 2687.0, 1883.0, 1210.0, 1201.0, 3169.0, 3098.0, 3688.0, 2409.0, 3198.0, 610.0, 1923.0, 1982.0, 165.0, 2403.0, 784.0, 2871.0, 2889.0, 628.0, 2300.0, 417.0, 3671.0, 3100.0, 3914.0, 3608.0, 3152.0, 3429.0, 1794.0, 952.0, 1391.0, 2518.0, 410.0, 3535.0, 2333.0, 1713.0, 2605.0, 707.0, 2795.0, 1965.0, 373.0, 3916.0, 556.0, 3703.0, 95.0, 466.0, 3066.0, 3177.0, 2088.0, 1476.0, 163.0, 3422.0, 58.0, 1244.0, 1689.0, 2002.0, 1711.0, 2259.0, 3524.0, 1371.0, 3104.0, 1693.0, 965.0, 1732.0, 2600.0, 3424.0, 3755.0, 2450.0, 3826.0, 3801.0, 3927.0, 1298.0, 2118.0, 112.0, 2478.0, 471.0, 1673.0, 1246.0, 2734.0, 2529.0, 2806.0, 1948.0, 2093.0, 45.0, 648.0, 3504.0, 2968.0, 1722.0, 1963.0, 2840.0, 1747.0, 1348.0, 3871.0, 3175.0, 2360.0, 1092.0, 3190.0, 1405.0, 367.0, 3248.0, 1702.0, 1734.0, 2644.0, 1597.0, 1401.0, 1416.0, 107.0, 1379.0, 2764.0, 2116.0, 1036.0, 60.0, 2115.0, 1876.0, 1254.0, 2243.0, 2606.0, 3925.0, 3087.0, 1627.0, 3770.0, 3678.0, 3113.0, 3036.0, 3525.0, 1584.0, 2236.0, 3267.0, 954.0, 1205.0, 2470.0, 2686.0, 3397.0, 2015.0, 1377.0, 3740.0, 1594.0, 2456.0, 2038.0, 891.0, 1342.0, 1966.0, 2808.0, 3324.0, 3794.0, 2467.0, 3420.0, 3773.0, 1927.0, 2231.0, 3742.0, 1960.0, 1542.0, 2672.0, 1376.0, 3174.0, 1248.0, 225.0, 1267.0, 3203.0, 1025.0, 2769.0, 1973.0, 2541.0, 3593.0, 2058.0, 3273.0, 154.0, 1179.0, 2009.0, 2423.0, 2676.0, 2793.0, 3505.0, 1920.0, 3357.0, 2580.0, 2542.0, 1701.0, 3252.0, 440.0, 540.0, 1885.0, 2384.0, 1414.0, 1251.0, 1187.0, 2841.0, 2287.0, 2004.0, 1257.0, 1358.0, 2253.0, 3918.0, 2976.0, 1100.0, 2140.0, 2092.0, 2772.0, 3500.0, 1196.0, 3728.0, 555.0, 3564.0, 3099.0, 2863.0, 2492.0, 13.0, 2378.0, 3271.0, 3946.0, 1017.0, 3189.0, 3908.0, 1238.0, 3551.0, 800.0, 1193.0, 3254.0, 3614.0, 448.0, 1779.0, 3477.0, 1388.0, 748.0, 1411.0, 3948.0, 1057.0, 2877.0, 2633.0, 3078.0, 2289.0, 514.0, 3831.0, 535.0, 361.0, 290.0, 1408.0, 1356.0, 2522.0, 2321.0, 1395.0, 1103.0, 2861.0, 1974.0, 2497.0, 1633.0, 2530.0, 1931.0, 125.0, 1735.0, 3159.0, 892.0, 2828.0, 523.0, 3148.0, 296.0, 2882.0, 1639.0, 1665.0, 3834.0, 534.0, 2942.0, 1247.0, 861.0, 2107.0, 3469.0, 1970.0, 3307.0, 432.0, 3879.0, 3930.0, 742.0, 3937.0, 1237.0, 1091.0, 3214.0, 1273.0, 3809.0, 3115.0, 2111.0, 468.0, 3769.0, 2961.0, 3771.0, 246.0, 3094.0, 2907.0, 1016.0, 151.0, 377.0, 450.0, 3538.0, 3717.0, 2694.0, 2745.0, 2389.0, 3865.0, 281.0, 2272.0, 2991.0, 1810.0, 2024.0, 2725.0, 2731.0, 409.0, 2971.0, 1083.0, 2701.0, 1753.0, 1459.0, 2567.0, 673.0, 3516.0, 611.0, 947.0, 1176.0, 1640.0, 172.0, 2671.0, 2041.0, 2723.0, 2471.0, 378.0, 3901.0, 1834.0, 1733.0, 1135.0, 998.0, 2475.0, 292.0, 3347.0, 2121.0, 3952.0, 1219.0, 413.0, 2294.0, 1997.0, 849.0, 2017.0, 2025.0, 3476.0, 1399.0, 2822.0, 2068.0, 180.0, 2076.0, 3700.0, 1783.0, 3326.0, 1760.0, 2437.0, 3893.0, 2594.0, 16.0, 1942.0, 2171.0, 2815.0, 1281.0, 1589.0, 936.0, 3168.0, 2520.0, 3095.0, 3448.0, 1971.0, 1230.0, 3129.0, 3799.0, 3125.0, 3784.0, 3789.0, 3262.0, 1946.0, 2390.0, 1918.0, 3201.0, 3909.0, 2943.0, 2082.0, 3157.0, 2112.0, 3409.0, 1772.0, 1680.0, 3633.0, 2153.0, 720.0, 674.0, 3713.0, 126.0, 585.0, 2353.0, 158.0, 3676.0, 3398.0, 485.0, 765.0, 1284.0, 2089.0, 1148.0, 1147.0, 2183.0, 1037.0, 2393.0, 2250.0, 2524.0, 1617.0, 1457.0, 3135.0, 3142.0, 2935.0, 1461.0, 533.0, 1425.0, 1282.0, 728.0, 3521.0, 1972.0, 1361.0, 551.0, 2016.0, 454.0, 3889.0, 3837.0, 190.0, 2735.0, 2124.0, 2310.0, 23.0, 3548.0, 1466.0, 3743.0, 1124.0, 2033.0, 1590.0, 2138.0, 2716.0, 1649.0, 1189.0, 2135.0, 3243.0, 3359.0, 1339.0, 123.0, 1224.0, 2996.0, 344.0, 1101.0, 515.0, 2428.0, 1873.0, 1392.0, 2583.0, 258.0, 2519.0, 2771.0, 213.0, 451.0, 2906.0, 2313.0, 3253.0, 1343.0, 2941.0, 745.0, 2729.0, 353.0, 1707.0, 2859.0, 2108.0, 1359.0]
        L_f = [920.0, 3844.0, 2369.0, 1088.0, 3534.0, 1207.0, 17.0, 1041.0, 3512.0, 3418.0, 1188.0, 902.0, 2336.0, 3911.0, 1441.0, 141.0, 2690.0, 928.0, 39.0, 2762.0, 906.0, 838.0, 2657.0, 2125.0, 3565.0, 1967.0, 2291.0, 914.0, 932.0, 1620.0, 2160.0, 247.0, 222.0, 261.0, 2881.0, 2145.0, 3072.0, 1028.0, 1956.0, 2080.0, 1286.0, 3798.0, 1959.0, 28.0, 2248.0, 3247.0, 3594.0, 3155.0, 1345.0, 531.0, 1277.0, 593.0, 3044.0, 3083.0, 3005.0, 1380.0, 2020.0, 105.0, 1678.0, 1608.0, 2572.0, 3791.0, 1104.0, 2144.0, 318.0, 1186.0, 1073.0, 595.0, 2724.0, 1641.0, 351.0, 2908.0, 357.0, 3079.0, 1688.0, 3556.0, 3186.0, 2406.0, 224.0, 1962.0, 1480.0, 3251.0, 11.0, 345.0, 3526.0, 1784.0, 951.0, 3668.0, 2485.0, 1958.0, 2739.0, 916.0, 950.0, 2443.0, 3684.0, 904.0, 898.0, 587.0, 552.0, 2143.0, 3481.0, 3097.0, 3067.0, 1449.0, 47.0, 616.0, 3281.0, 1259.0, 661.0, 2348.0, 562.0, 3606.0, 2496.0, 2085.0, 1271.0, 372.0, 2857.0, 3325.0, 1394.0, 1081.0, 1032.0, 918.0, 1409.0, 314.0, 899.0, 733.0, 2245.0, 381.0, 2316.0, 232.0, 2405.0, 2677.0, 1066.0, 2396.0, 2282.0, 1059.0, 2622.0, 1941.0, 959.0, 3479.0, 3124.0, 1197.0, 1777.0, 915.0, 955.0, 1648.0, 3705.0, 3061.0, 34.0, 1285.0, 1.0, 2875.0, 1150.0, 3545.0, 2664.0, 2155.0, 1097.0, 262.0, 3915.0, 971.0, 2186.0, 3702.0, 3105.0, 2280.0, 3604.0, 3515.0, 1513.0, 2331.0, 1500.0, 2803.0, 945.0, 2639.0, 3051.0, 837.0, 3408.0, 457.0, 1801.0, 2506.0, 4.0, 2469.0, 270.0, 46.0, 1235.0, 2355.0, 2346.0, 1357.0, 461.0, 3255.0, 3176.0, 3350.0, 2975.0, 2014.0, 3936.0, 2072.0, 1353.0, 2006.0, 1397.0, 2612.0, 1099.0, 1367.0, 3270.0, 938.0, 2357.0, 94.0, 412.0, 1518.0, 3591.0, 538.0, 2000.0, 2846.0, 708.0, 329.0, 2995.0, 653.0, 1280.0, 5.0, 337.0, 1022.0, 2468.0, 1569.0, 905.0, 1031.0, 900.0, 1541.0, 2926.0, 3730.0, 1900.0, 2718.0, 1021.0, 3185.0, 2746.0, 327.0, 2805.0, 3101.0, 2920.0, 3269.0, 1674.0, 477.0, 3686.0, 2077.0, 2801.0, 581.0, 2133.0, 3724.0, 3296.0, 3554.0, 3478.0, 1479.0, 3720.0, 491.0, 1014.0, 1236.0, 3134.0, 695.0, 2763.0, 1013.0, 1096.0, 1856.0, 2827.0, 248.0, 1875.0, 3211.0, 3672.0, 215.0, 3224.0, 3396.0, 469.0, 1897.0, 3528.0, 2870.0, 917.0, 930.0, 1654.0, 3328.0, 3786.0, 907.0, 3870.0, 1422.0, 2206.0, 2114.0, 2324.0, 2575.0, 919.0, 3467.0, 1047.0, 1806.0, 350.0, 230.0, 2505.0, 48.0, 182.0, 144.0, 170.0, 2141.0, 1916.0, 3081.0, 1191.0, 1086.0, 2598.0, 546.0, 1407.0, 153.0, 2635.0, 2057.0, 2037.0, 1327.0, 3145.0, 446.0, 2193.0, 1337.0, 1913.0, 195.0, 2132.0, 1804.0, 3562.0, 3706.0, 1172.0, 1042.0, 2946.0, 2514.0, 1093.0, 1616.0, 3011.0, 2151.0, 1111.0, 613.0, 1043.0, 2774.0, 2154.0, 2621.0, 52.0, 3060.0, 3723.0, 206.0, 3133.0, 1821.0, 1964.0, 211.0, 2454.0, 532.0, 218.0, 3156.0, 1586.0, 1126.0, 2096.0, 927.0, 2007.0, 778.0, 2097.0, 3117.0, 691.0, 3567.0, 1223.0, 1268.0, 1300.0, 2747.0, 1573.0, 3302.0, 671.0, 3471.0, 3825.0, 1064.0, 1299.0, 252.0, 3004.0, 2091.0, 2337.0, 61.0, 1020.0, 3763.0, 1727.0, 74.0, 3599.0, 3708.0, 465.0, 29.0, 3741.0, 3457.0, 2399.0, 781.0, 69.0, 3635.0, 3808.0, 3249.0, 2732.0, 1621.0, 1686.0, 3435.0, 3857.0, 3299.0, 3426.0, 176.0, 343.0, 2972.0, 2853.0, 272.0, 2788.0, 1393.0, 203.0, 1465.0, 801.0, 1917.0, 2431.0, 3714.0, 2967.0, 3553.0, 79.0, 3951.0, 1683.0, 3071.0, 3102.0, 302.0, 3655.0, 2261.0, 3877.0, 2266.0, 3716.0, 3699.0, 1769.0, 266.0, 1173.0, 2693.0, 3093.0, 1658.0, 277.0, 279.0, 848.0, 839.0, 2365.0, 2738.0, 1264.0, 271.0, 1269.0, 2043.0, 3855.0, 1030.0, 1346.0, 2052.0, 2142.0, 2719.0, 2574.0, 2053.0, 1410.0, 3912.0, 1381.0, 3660.0, 2446.0, 2613.0, 2314.0, 978.0, 348.0, 2168.0, 3466.0, 669.0, 3649.0, 2448.0, 2899.0, 1611.0, 2940.0, 8.0, 1463.0, 26.0, 3557.0, 1994.0, 1758.0, 414.0, 1027.0, 3088.0, 3391.0, 1936.0, 2205.0, 3861.0, 332.0, 3450.0, 2585.0, 3618.0, 425.0, 1605.0, 3827.0, 846.0, 2267.0, 2359.0, 2952.0, 2786.0, 3923.0, 1290.0, 3240.0, 3388.0, 1547.0, 338.0, 3712.0, 3063.0, 242.0, 715.0, 3679.0, 3571.0, 668.0, 1069.0, 2276.0, 1438.0, 2688.0, 2900.0, 168.0, 3539.0, 199.0, 3675.0, 2436.0, 647.0, 724.0, 82.0, 542.0, 1362.0, 117.0, 2109.0, 3246.0, 3019.0, 1904.0, 360.0, 2966.0, 482.0, 2741.0, 334.0, 2100.0, 2173.0, 1615.0, 358.0, 280.0, 3932.0, 369.0, 3547.0, 3739.0, 1788.0, 875.0, 2106.0, 3719.0, 3839.0, 1417.0, 3566.0, 3795.0, 670.0, 520.0, 208.0, 3449.0, 3274.0, 27.0, 3872.0, 2969.0, 2927.0, 2442.0, 113.0, 2084.0, 1848.0, 3882.0, 3790.0, 3926.0, 2820.0, 3922.0, 3046.0, 832.0, 3896.0, 2101.0, 1600.0, 2548.0, 2453.0, 386.0, 239.0, 1015.0, 85.0, 3077.0, 3264.0, 3340.0, 3114.0, 1729.0, 2498.0, 309.0, 1034.0, 2421.0, 3438.0, 2599.0, 405.0, 3461.0, 3813.0, 3238.0, 3399.0, 3921.0, 912.0, 1840.0, 2876.0, 319.0, 40.0, 257.0, 3287.0, 880.0, 754.0, 1874.0, 2241.0, 2553.0, 1699.0, 550.0, 1549.0, 2338.0, 1922.0, 3612.0, 1894.0, 1049.0, 1185.0, 2779.0, 3902.0, 3580.0, 2.0, 2435.0, 73.0, 1012.0, 1275.0, 783.0, 512.0, 1919.0, 3838.0, 2903.0, 507.0, 1896.0, 2263.0, 2320.0, 1515.0, 363.0, 3492.0, 1562.0, 1588.0, 408.0, 3405.0, 307.0, 1199.0, 3268.0, 186.0, 1961.0, 1428.0, 2540.0, 3284.0, 2062.0, 3624.0, 1169.0, 2513.0, 575.0, 380.0, 2696.0, 2070.0, 2130.0, 3897.0, 615.0, 50.0, 3852.0, 415.0, 1797.0, 1660.0, 506.0, 3704.0, 2816.0, 2678.0, 2122.0, 1836.0, 2126.0, 481.0, 87.0, 3577.0, 2990.0, 3200.0, 441.0, 1554.0, 346.0, 1653.0, 2202.0, 2616.0, 283.0, 3584.0, 2417.0, 2284.0, 2042.0, 3454.0, 1582.0, 2568.0, 1669.0, 2048.0, 3613.0, 1911.0, 949.0, 420.0, 1719.0, 2361.0, 41.0, 3949.0, 379.0, 2379.0, 3447.0, 2136.0, 2642.0, 3206.0, 1995.0, 3150.0, 2856.0, 2010.0, 2532.0, 382.0, 2398.0, 1798.0, 1242.0, 2414.0, 2550.0, 1084.0, 131.0, 3055.0, 2630.0, 1949.0, 1954.0, 2352.0, 2110.0, 3181.0, 2021.0, 1344.0, 3685.0, 1398.0, 1312.0, 910.0, 3738.0, 173.0, 1456.0, 3445.0, 986.0, 2848.0, 2722.0, 3696.0, 3864.0, 3707.0, 1171.0, 558.0, 356.0, 2717.0, 3204.0, 2561.0, 934.0, 2704.0, 371.0, 1831.0, 879.0, 2439.0, 3108.0, 2517.0, 1372.0, 1672.0, 807.0, 3616.0, 688.0, 2797.0, 519.0, 1211.0, 1730.0, 1446.0, 1546.0, 2445.0, 2147.0, 3475.0, 1556.0, 1580.0, 1220.0, 2373.0, 501.0, 124.0, 1216.0, 1429.0, 2683.0, 2066.0, 1881.0, 2949.0, 3090.0, 802.0, 1870.0, 407.0, 586.0, 1944.0, 2989.0, 1921.0, 1226.0, 2380.0, 3489.0, 3886.0, 2190.0, 2919.0, 2495.0, 2392.0, 753.0, 1484.0, 1667.0, 2363.0, 3308.0, 1077.0, 1805.0, 2714.0, 3173.0, 216.0, 1694.0, 736.0, 1321.0, 1483.0, 608.0, 1485.0, 1347.0, 2789.0, 25.0, 2699.0, 1792.0, 2065.0, 2709.0, 2860.0, 1845.0, 2752.0, 494.0, 2273.0, 62.0, 2710.0, 866.0, 3841.0, 1566.0, 3153.0, 973.0, 3600.0, 1240.0, 1270.0, 923.0, 2159.0, 896.0, 3258.0, 147.0, 3439.0, 2947.0, 2643.0, 1212.0, 1258.0, 2527.0, 1419.0, 1217.0, 316.0, 1293.0, 2420.0, 3130.0, 2474.0, 2879.0, 991.0, 3317.0, 2713.0, 3440.0, 2463.0, 1619.0, 2539.0, 3070.0, 3040.0, 2163.0, 508.0, 428.0, 1816.0, 2533.0, 2736.0, 1969.0, 3054.0, 2176.0, 288.0, 2794.0, 2239.0, 2290.0, 1234.0, 3735.0, 2166.0, 19.0, 2071.0, 2394.0, 2858.0]
        # --- for this data need to find popular lm, lf
    elif dataset == 'yahoo':
        X = DL.load_user_item_matrix_yahoo()
        T = DL.load_gender_vector_yahoo()
        # updated list with each users min rating 20
        L_m = [2587, 3581, 4289, 4049, 132, 916, 7038, 1111, 6790, 1691, 372, 5818, 7266, 1946, 3713, 7661, 2450, 6177, 1487, 4249, 6787, 6262, 4743, 6590, 7262, 8346, 7565, 5073, 5061, 5003, 1442, 7660, 1409, 7064, 2956, 7451, 3425, 1367, 5300, 5908, 7063, 2858, 3210, 292, 7288, 6750, 3123, 4507, 1278, 5373, 5040, 1134, 7895, 6763, 6539, 1483, 2802, 2998, 1066, 4016, 6547, 5164, 3471, 1430, 5532, 1556, 1106, 3239, 3887, 4217, 1415, 7558, 3582, 3534, 6574, 4343, 5729, 762, 6635, 4639, 802, 8568, 3948, 3724, 5577, 4789, 3326, 4481, 6185, 1165, 6811, 5592, 1615, 3755, 6376, 2590, 3258, 6582, 5582, 1376, 1799, 3199, 1555, 5227, 4358, 5265, 4522, 144, 6858, 8287, 1863, 6925, 6292, 6412, 6482, 4004, 5216, 7220, 7759, 2686, 2925, 5130, 2368, 177, 2366, 5013, 3249, 3245, 5937, 578, 2260, 984, 1351, 8141, 3940, 5555, 2115, 4459, 8315, 2693, 1867, 4252, 8136, 3153, 3186, 4056, 3487, 1947, 5935, 769, 1744, 6789, 5814, 4962, 6116, 2677, 8529, 4870, 3570, 6718, 4068, 2947, 1805, 5043, 6455, 6992, 6067, 2930, 3394, 6270, 4244, 7601, 8464, 2648, 5796, 6165, 2815, 5972, 6753, 6857, 3317, 3630, 327]
        L_f = [8569, 8176, 8494, 5099, 8218, 8533, 4931, 126, 760, 7813, 8563, 4468, 8219, 562, 8319, 4636, 1100, 8215, 8379, 1642, 8072, 8323, 3618, 7020, 7864, 7628, 4804, 441, 323, 719, 5302, 7885, 8390, 2315, 8306, 8238, 8301, 8253, 1160, 2405, 1970, 8177, 6944, 5675, 8093, 7656, 1576, 1362, 550, 4819, 6957, 939, 4234, 2258, 6970, 5448, 352, 7651, 7490, 8349, 7600, 54, 7781, 6221, 100, 7478, 92, 8430, 7081, 7587, 5039, 4233, 7592, 2972, 7498, 8506, 4903, 7778, 282, 8235, 6801, 6357, 8474, 303, 7972, 7630, 1621, 6948, 5984, 5391, 52, 36, 6991, 4464, 4893, 7883, 8039, 423, 7732, 3964, 291, 8531, 235, 5225, 1971, 1292, 4280, 5291, 7589, 210, 654, 361, 7557, 8459, 7834, 8134, 6932, 8227, 1101, 587, 7983, 4274, 606, 6967, 7005, 634, 7590, 110, 5841, 7860, 5521, 215, 4010, 542, 7996, 7466, 7990, 7644, 8418, 616, 8425, 8470, 8033, 388, 7756, 382, 5967, 7769, 4486, 5464, 7768, 384, 7705, 6761, 8370, 1908, 8092, 8318, 8398, 5825, 6937, 7772, 8362, 7703, 485, 7835, 8123, 5443, 2023, 8165, 5623, 7737, 5890, 8249, 2906, 4629, 8188, 149, 468, 407, 7987, 4892, 8003, 7964, 5376, 5687, 7655, 563, 6910, 4963, 7999, 7796, 8041, 4741, 4203, 4699, 8485, 6895, 5529, 6193, 7896, 597, 5159, 8027, 7479, 818, 7798, 6587, 601, 3807, 3, 8572, 6904, 2052, 621, 8266, 5850, 7483, 8048, 3941, 8486, 5404, 7936, 5134, 8303, 325, 3831, 8057, 6405, 8157, 373, 3013, 5621, 87, 6894, 8071, 5614, 5605, 8091, 8274, 8206, 329, 6488, 6837, 3826, 2323, 5025, 2494, 8058, 62, 3071, 6174, 5884, 5838, 7707, 5865, 8070, 356, 1250, 5539, 125, 7240, 7949, 3859, 182, 69, 4271, 8481, 8061, 5630, 7854, 3509, 7958, 10, 7502, 4525, 8083, 6462, 7873, 557, 154, 137, 5956, 7809, 8180, 6105, 3357, 5307, 5485, 1285, 6343, 7612, 8046, 7459, 6922, 70, 1713, 855, 8438, 204, 7945, 2085, 4337, 7747, 7633, 7941, 7012, 7843, 2567, 2522, 1872, 620, 558, 8226, 4851, 7973, 8233, 527, 7697, 213, 8214, 8421, 541, 8240, 4466, 6756, 7912, 8060, 5421, 5665, 2155, 3881, 7994, 3195, 7131, 8205, 2555, 6930, 7539, 4152, 7731, 3697, 7968, 607, 271, 5809, 239, 11, 4424, 3692, 8232, 7, 6239, 8295, 3073, 365, 5974, 1922, 1136, 2986, 7853, 2227, 7795, 6839, 1213, 8239, 524, 8124, 5579, 5530, 276, 519, 4813, 5552, 5034, 7704, 7788, 8162, 5308, 2234, 1416, 8472, 8225, 3292, 7842, 8154, 6999, 5342, 4514, 7634, 7613, 820, 8095, 3794, 6161, 102, 8327, 253, 7810, 7443, 1043, 107, 3949, 7741, 25, 254, 4981, 7519, 411, 3838, 7829, 8155, 8208, 5768, 8312, 3293, 6015, 7525, 8203, 71, 6997, 8244, 8326, 5755, 7579, 8343, 2017, 2386, 6849, 5052, 6091, 2583, 2088, 422, 8490, 4769, 469, 514, 6897, 575, 763, 3314, 3883, 8160, 8189, 5672, 7211, 5553, 6266, 8211, 7706, 8350, 805, 455, 5058, 4900, 403, 7775, 7876, 7461, 217, 6908, 8037, 7620, 7982, 1785, 7955, 8434, 8289, 7098, 6018, 5919, 5878, 5245, 8537, 8273, 538, 7838, 3614, 8212, 8193, 8210, 13, 8030, 5800, 842, 8250, 8269, 3580, 6211, 7520, 566, 8143, 8373, 6444, 3337, 8159, 4258, 7708, 8204, 7848, 7567, 151, 7937, 694, 473, 8190, 248, 8264]

    # outside loop

    # Diagnostics: Print the number of users in X
    print("Number of users in X:", X.shape[0])

    # Ensure len_dict contains all user indices
    for i in range(X.shape[0]):
        if i not in len_dict:
            len_dict[i] = []

    # 1: get the set of most correlated movies, L_f and L_m:
    k = 100
    # k = 50
    L_m = list(map(lambda x: x-1, L_m))[:k]
    L_f = list(map(lambda x: x-1, L_f))[:k]

    print(len(len_dict))
    for z in range(len(X)):
        #print(z)
        values = len_dict[z]
        lst_j = []
        # list of neighbors ordered / ranked by weight for user i
        user_item = list(np.argsort(values))  # [::-1])

        if (len(user_item) == len(values)):
            p = 0
            while p < len(values):
                if T[z] == 0:
                    f = user_item.pop(0)  # np.argmin (lst)
                    print(f)
                    if f in L_f:
                        if f not in lst_j:
                            lst_j.append(f)

                elif T[z] == 1:
                    m = user_item.pop(0)
                    print(m)
                    if m in L_m:
                        if m not in lst_j:
                            lst_j.append(m)
                p += 1
            item_choice[z] = lst_j
    print("item_choice: ", item_choice)
    with open("ml-"+dataset+"/NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top"+str(k)+"IndicativeItems_noRemoval.json",
              "w") as fp:
        json.dump(item_choice, fp, cls=NpEncoder)

# -------------------------------- do later for top50. -> now regen for ml-100k


"""PerBlur without removal strategy function for obfuscating the user-item matrix"""

def PerBlur_No_Removal():

    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    top = 100
    p = 0.01
    notice_factor = 2
    dataset = ['100k', '1m', 'yahoo'][2]

    if dataset == '100k':
        X = DL.load_user_item_matrix_100k()
        T = DL.load_gender_vector_100k()
        X_filled = DL.load_user_item_matrix_100k_Impute()

    elif dataset == '1m':
        X = DL.load_user_item_matrix_1m()
        T = DL.load_gender_vector_1m()
        X_filled = DL.load_user_item_matrix_1m_Impute()

    elif dataset == 'yahoo':
        X = DL.load_user_item_matrix_yahoo()
        T = DL.load_gender_vector_yahoo()
        X_filled = DL.load_user_item_matrix_yahoo_Impute()

    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = [rating for rating in X[:, item_id] if rating > 0 ]
        avg_ratings[item_id] = np.average(ratings) if ratings else 0
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor

    # 1: get the set of most correlated movies, L_f and L_m:
    with open('ml-'+dataset+'/test_NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top100IndicativeItems_noRemoval.json') as json_file:
        item_choice = json.load(json_file)

    # Now, where we have the two lists, we can start obfuscating the data:
    X_obf = np.copy(X)
    total_added = 0
    for index, user in enumerate(X):
        rate = 0
        for rating in user:
            if rating > 0:
                rate +=1
        k = rate * p
        greedy_index = 0
        added = 0
        mylist = list(item_choice.values())
        safety_counter = 0
        print(f"User: {index}, {user} and no of rating: {rate}, p:{p} & k = {k}")
        while added < k and safety_counter < 100: # 1000
            if greedy_index >= len(mylist[index]):
                safety_counter = 100
                continue
            if sample_mode == 'greedy':
                vec = mylist[index]
                movie_id = vec[greedy_index]
                movie_id = movie_id
            if sample_mode == 'random':
                movie_id = vec[np.random.randint(0, len(vec))]
                movie_id = int(movie_id)
            greedy_index += 1

            rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, movie_id]])
            if rating_count > max_count[movie_id]:
                continue

            if X_obf[index, movie_id] == 0:# and X_test [index, int(movie_id) ] == 0:
                X_obf[index, movie_id] =  avg_ratings[int(movie_id)] # X_filled[index, movie_id] #  #
                print(f"obf: {X_obf[index, movie_id]}, movie: {movie_id}")
                added += 1
            safety_counter += 1
        total_added += added
        print(f"user: {index}, added items: {added}, total_added: {total_added}")

    # Save the obfuscated data to a file
    output_file = 'ml-'+dataset+'/PerBlur/'
    print(output_file)
    with open(output_file + "PerBlur_" + rating_mode + "_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(int(np.round(rating))) + "::000000000\n")

    return X_obf

#-----------------------------------------------

"""PerBlur with removal function for obfuscatibg the user-item matrix"""


def PerBlur():

    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    removal_mode = list(['random', 'strategic'])[1]
    top = 100
    p = 0.05
    notice_factor = 2

    dataset = ['100k', '1m', 'yahoo'][2]

    if dataset == '100k':
        X = DL.load_user_item_matrix_100k()
        T = DL.load_gender_vector_100k()
        X_filled = DL.load_user_item_matrix_100k_Impute()
        L_m = [751, 327, 24, 688, 472, 888, 317, 948, 315, 178, 305, 179, 359, 186, 10, 883, 294, 342, 5, 471, 1048, 1101, 879, 449, 324, 150, 357, 411, 1265, 136, 524, 678, 202, 692, 100, 303, 264, 339, 1010, 276, 218, 995, 240, 433, 271, 203, 544, 520, 410, 222, 1028, 924, 823, 322, 462, 265, 750, 23, 281, 137, 333, 736, 164, 358, 301, 479, 647, 168, 144, 94, 687, 60, 70, 175, 6, 354, 344, 1115, 597, 596, 42, 14, 521, 184, 393, 188, 450, 1047, 682, 7, 760, 737, 270, 1008, 260, 295, 1051, 289, 430, 235, 258, 330, 56, 347, 52, 205, 371, 1024, 8, 369, 116, 79, 831, 694, 287, 435, 495, 242, 33, 201, 230, 506, 979, 620, 355, 181, 717, 936, 500, 893, 489, 1316, 855, 2, 180, 871, 755, 523, 477, 227, 412, 183, 657, 510, 77, 474, 1105, 67, 705, 362, 1133, 511, 546, 768, 96, 457, 108, 1137, 352, 341, 825, 165, 584, 505, 908, 404, 679, 15, 513, 117, 490, 665, 28, 338, 59, 32, 1014, 989, 351, 475, 57, 864, 969, 177, 316, 463, 134, 703, 306, 378, 105, 99, 229, 484, 651, 157, 232, 114, 161, 395, 754, 931, 591, 408, 685, 97, 554, 468, 455, 640, 473, 683, 300, 31, 1060, 650, 72, 191, 259, 1280, 199, 826, 747, 68, 1079, 887, 578, 329, 274, 174, 428, 905, 780, 753, 396, 4, 277, 71, 519, 1062, 189, 325, 502, 233, 1022, 880, 1063, 197, 813, 16, 331, 208, 162, 11, 963, 501, 820, 930, 896, 318, 1142, 1194, 425, 171, 282, 496, 26, 215, 573, 527, 730, 49, 693, 517, 336, 417, 207, 900, 299, 226, 606, 268, 192, 407, 62, 636, 480, 1016, 241, 945, 343, 603, 231, 469, 515, 492, 664, 972, 642, 781, 984, 187, 149, 257, 40, 141, 978, 44, 845, 85, 244, 512, 182, 1021, 756, 1, 778, 644, 1050, 185, 530, 81, 497, 3, 335, 923, 509, 418, 724, 566, 221, 570, 655, 413, 1311, 340, 583, 537, 635, 686, 176, 296, 926, 561, 101, 173, 862, 680, 652]
        L_f = [332, 321, 877, 906, 827, 882, 337, 278, 872, 304, 87, 93, 1038, 292, 629, 727, 220, 898, 107, 143, 516, 819, 382, 499, 876, 45, 1600, 111, 416, 132, 752, 125, 689, 604, 628, 310, 246, 419, 148, 659, 881, 248, 899, 707, 20, 129, 272, 690, 350, 498, 19, 216, 713, 429, 253, 372, 311, 251, 288, 1023, 392, 1328, 131, 320, 83, 895, 912, 445, 1094, 432, 109, 156, 532, 588, 486, 653, 243, 319, 518, 269, 298, 447, 290, 1386, 988, 423, 723, 279, 696, 568, 919, 1296, 266, 645, 937, 595, 482, 748, 348, 749, 815, 451, 154, 684, 993, 904, 204, 198, 403, 525, 553, 671, 507, 1119, 1036, 65, 420, 169, 9, 427, 832, 894, 729, 654, 1128, 987, 1281, 401, 1089, 942, 190, 406, 170, 402, 878, 291, 710, 297, 721, 151, 367, 954, 856, 64, 312, 385, 581, 370, 39, 648, 691, 746, 89, 1033, 745, 194, 662, 145, 539, 98, 147, 739, 159, 90, 236, 55, 120, 448, 200, 409, 623, 118, 731, 365, 238, 124, 146, 716, 1026, 847, 261, 153, 262, 130, 127, 866, 58, 356, 195, 1086, 1190, 1041, 1073, 889, 405, 950, 891, 611, 582, 66, 1035, 346, 1394, 172, 909, 1013, 785, 925, 488, 704, 1315, 708, 1431, 309, 121, 239, 902, 213, 446, 1012, 1054, 873, 193, 478, 829, 916, 353, 349, 133, 249, 550, 638, 452, 1313, 466, 529, 458, 792, 313, 625, 668, 1061, 742, 1065, 763, 73, 846, 1090, 863, 1620, 890, 1176, 1017, 237, 1294, 245, 508, 387, 53, 514, 422, 1068, 1527, 939, 1232, 1011, 631, 381, 956, 875, 459, 607, 1442, 155, 697, 1066, 1285, 293, 88, 1221, 1109, 675, 254, 209, 649, 140, 48, 613, 962, 1157, 991, 1083, 720, 535, 901, 436, 286, 559, 1085, 892, 102, 211, 285, 1383, 758, 1234, 674, 1163, 283, 637, 735, 885, 630, 709, 938, 38, 142, 219, 953, 275, 1059, 676, 210, 47, 63, 255, 494, 744, 250, 1114, 672, 308, 632, 17, 1203, 762, 1522, 538, 491, 307, 669, 775, 1135, 217, 841, 949, 766, 966, 314, 812, 821, 574, 1039, 1388, 51, 579, 252, 50, 1009, 1040, 1147, 224, 212, 1483, 1278, 122, 934, 702, 443, 86, 1444, 69, 400, 975, 35, 549, 928, 531, 622, 614, 1020, 658, 740, 1468, 22, 1007, 470, 1269, 633, 1074, 1299, 1095, 1148, 280, 1136, 92, 167, 434, 284, 961, 1084, 126, 619, 974, 196, 485, 1152, 673, 627, 345, 29, 605, 929, 952, 714, 1053, 526, 123, 476, 660, 955, 380, 1503, 163, 493, 1197, 431, 504, 886, 1037, 13, 794, 273, 844, 1032, 1025, 106, 91, 533, 421, 699, 869, 78, 1243, 481, 661, 82, 732]

    elif dataset == '1m':
        X = DL.load_user_item_matrix_1m()
        T = DL.load_gender_vector_1m()
        X_filled = DL.load_user_item_matrix_1m_Impute()
        L_m = [589.0, 1233.0, 2706.0, 1090.0, 2959.0, 1250.0, 2427.0, 2490.0, 1208.0, 1266.0, 3654.0, 1748.0, 1262.0, 1396.0, 1374.0, 2700.0, 1265.0, 1089.0, 1222.0, 231.0, 2770.0, 1676.0, 2890.0, 1228.0, 1136.0, 3360.0, 3298.0, 1663.0, 3811.0, 2011.0, 1261.0, 233.0, 3361.0, 2366.0, 1127.0, 1276.0, 3555.0, 1214.0, 3929.0, 299.0, 1304.0, 3468.0, 1095.0, 150.0, 1213.0, 750.0, 3082.0, 6.0, 111.0, 3745.0, 349.0, 541.0, 2791.0, 785.0, 1060.0, 1294.0, 1302.0, 1256.0, 1292.0, 2948.0, 3683.0, 3030.0, 3836.0, 913.0, 2150.0, 32.0, 2826.0, 2721.0, 590.0, 3623.0, 2997.0, 3868.0, 3147.0, 1610.0, 3508.0, 2046.0, 21.0, 1249.0, 10.0, 1283.0, 3760.0, 2712.0, 3617.0, 3552.0, 3256.0, 1079.0, 3053.0, 1517.0, 2662.0, 1953.0, 2670.0, 3578.0, 2371.0, 3334.0, 2502.0, 2278.0, 364.0, 3462.0, 2401.0, 3163.0, 2311.0, 852.0, 2916.0, 1378.0, 3384.0, 524.0, 70.0, 370.0, 3035.0, 3513.0, 2917.0, 3697.0, 24.0, 1957.0, 3494.0, 1912.0, 3752.0, 2013.0, 3452.0, 3928.0, 2987.0, 431.0, 2759.0, 1387.0, 1882.0, 3638.0, 1288.0, 2867.0, 2728.0, 2433.0, 161.0, 3386.0, 517.0, 741.0, 1287.0, 1231.0, 3062.0, 2288.0, 3753.0, 529.0, 3793.0, 3052.0, 2447.0, 1320.0, 3819.0, 1303.0, 922.0, 3022.0, 260.0, 858.0, 493.0, 3006.0, 480.0, 2410.0, 333.0, 1178.0, 3814.0, 2702.0, 1203.0, 2922.0, 1625.0, 3366.0, 3213.0, 2188.0, 2628.0, 3358.0, 2648.0, 3788.0, 953.0, 999.0, 3754.0, 3910.0, 3016.0, 3863.0, 303.0, 3263.0, 1080.0, 786.0, 3764.0, 2105.0, 3543.0, 2607.0, 3681.0, 592.0, 145.0, 2303.0, 1682.0, 1019.0, 3646.0, 1544.0, 235.0, 908.0, 3615.0, 2792.0, 2167.0, 2455.0, 1587.0, 1227.0, 2901.0, 2687.0, 1883.0, 1210.0, 1201.0, 3169.0, 3098.0, 3688.0, 2409.0, 3198.0, 610.0, 1923.0, 1982.0, 165.0, 2403.0, 784.0, 2871.0, 2889.0, 628.0, 2300.0, 417.0, 3671.0, 3100.0, 3914.0, 3608.0, 3152.0, 3429.0, 1794.0, 952.0, 1391.0, 2518.0, 410.0, 3535.0, 2333.0, 1713.0, 2605.0, 707.0, 2795.0, 1965.0, 373.0, 3916.0, 556.0, 3703.0, 95.0, 466.0, 3066.0, 3177.0, 2088.0, 1476.0, 163.0, 3422.0, 58.0, 1244.0, 1689.0, 2002.0, 1711.0, 2259.0, 3524.0, 1371.0, 3104.0, 1693.0, 965.0, 1732.0, 2600.0, 3424.0, 3755.0, 2450.0, 3826.0, 3801.0, 3927.0, 1298.0, 2118.0, 112.0, 2478.0, 471.0, 1673.0, 1246.0, 2734.0, 2529.0, 2806.0, 1948.0, 2093.0, 45.0, 648.0, 3504.0, 2968.0, 1722.0, 1963.0, 2840.0, 1747.0, 1348.0, 3871.0, 3175.0, 2360.0, 1092.0, 3190.0, 1405.0, 367.0, 3248.0, 1702.0, 1734.0, 2644.0, 1597.0, 1401.0, 1416.0, 107.0, 1379.0, 2764.0, 2116.0, 1036.0, 60.0, 2115.0, 1876.0, 1254.0, 2243.0, 2606.0, 3925.0, 3087.0, 1627.0, 3770.0, 3678.0, 3113.0, 3036.0, 3525.0, 1584.0, 2236.0, 3267.0, 954.0, 1205.0, 2470.0, 2686.0, 3397.0, 2015.0, 1377.0, 3740.0, 1594.0, 2456.0, 2038.0, 891.0, 1342.0, 1966.0, 2808.0, 3324.0, 3794.0, 2467.0, 3420.0, 3773.0, 1927.0, 2231.0, 3742.0, 1960.0, 1542.0, 2672.0, 1376.0, 3174.0, 1248.0, 225.0, 1267.0, 3203.0, 1025.0, 2769.0, 1973.0, 2541.0, 3593.0, 2058.0, 3273.0, 154.0, 1179.0, 2009.0, 2423.0, 2676.0, 2793.0, 3505.0, 1920.0, 3357.0, 2580.0, 2542.0, 1701.0, 3252.0, 440.0, 540.0, 1885.0, 2384.0, 1414.0, 1251.0, 1187.0, 2841.0, 2287.0, 2004.0, 1257.0, 1358.0, 2253.0, 3918.0, 2976.0, 1100.0, 2140.0, 2092.0, 2772.0, 3500.0, 1196.0, 3728.0, 555.0, 3564.0, 3099.0, 2863.0, 2492.0, 13.0, 2378.0, 3271.0, 3946.0, 1017.0, 3189.0, 3908.0, 1238.0, 3551.0, 800.0, 1193.0, 3254.0, 3614.0, 448.0, 1779.0, 3477.0, 1388.0, 748.0, 1411.0, 3948.0, 1057.0, 2877.0, 2633.0, 3078.0, 2289.0, 514.0, 3831.0, 535.0, 361.0, 290.0, 1408.0, 1356.0, 2522.0, 2321.0, 1395.0, 1103.0, 2861.0, 1974.0, 2497.0, 1633.0, 2530.0, 1931.0, 125.0, 1735.0, 3159.0, 892.0, 2828.0, 523.0, 3148.0, 296.0, 2882.0, 1639.0, 1665.0, 3834.0, 534.0, 2942.0, 1247.0, 861.0, 2107.0, 3469.0, 1970.0, 3307.0, 432.0, 3879.0, 3930.0, 742.0, 3937.0, 1237.0, 1091.0, 3214.0, 1273.0, 3809.0, 3115.0, 2111.0, 468.0, 3769.0, 2961.0, 3771.0, 246.0, 3094.0, 2907.0, 1016.0, 151.0, 377.0, 450.0, 3538.0, 3717.0, 2694.0, 2745.0, 2389.0, 3865.0, 281.0, 2272.0, 2991.0, 1810.0, 2024.0, 2725.0, 2731.0, 409.0, 2971.0, 1083.0, 2701.0, 1753.0, 1459.0, 2567.0, 673.0, 3516.0, 611.0, 947.0, 1176.0, 1640.0, 172.0, 2671.0, 2041.0, 2723.0, 2471.0, 378.0, 3901.0, 1834.0, 1733.0, 1135.0, 998.0, 2475.0, 292.0, 3347.0, 2121.0, 3952.0, 1219.0, 413.0, 2294.0, 1997.0, 849.0, 2017.0, 2025.0, 3476.0, 1399.0, 2822.0, 2068.0, 180.0, 2076.0, 3700.0, 1783.0, 3326.0, 1760.0, 2437.0, 3893.0, 2594.0, 16.0, 1942.0, 2171.0, 2815.0, 1281.0, 1589.0, 936.0, 3168.0, 2520.0, 3095.0, 3448.0, 1971.0, 1230.0, 3129.0, 3799.0, 3125.0, 3784.0, 3789.0, 3262.0, 1946.0, 2390.0, 1918.0, 3201.0, 3909.0, 2943.0, 2082.0, 3157.0, 2112.0, 3409.0, 1772.0, 1680.0, 3633.0, 2153.0, 720.0, 674.0, 3713.0, 126.0, 585.0, 2353.0, 158.0, 3676.0, 3398.0, 485.0, 765.0, 1284.0, 2089.0, 1148.0, 1147.0, 2183.0, 1037.0, 2393.0, 2250.0, 2524.0, 1617.0, 1457.0, 3135.0, 3142.0, 2935.0, 1461.0, 533.0, 1425.0, 1282.0, 728.0, 3521.0, 1972.0, 1361.0, 551.0, 2016.0, 454.0, 3889.0, 3837.0, 190.0, 2735.0, 2124.0, 2310.0, 23.0, 3548.0, 1466.0, 3743.0, 1124.0, 2033.0, 1590.0, 2138.0, 2716.0, 1649.0, 1189.0, 2135.0, 3243.0, 3359.0, 1339.0, 123.0, 1224.0, 2996.0, 344.0, 1101.0, 515.0, 2428.0, 1873.0, 1392.0, 2583.0, 258.0, 2519.0, 2771.0, 213.0, 451.0, 2906.0, 2313.0, 3253.0, 1343.0, 2941.0, 745.0, 2729.0, 353.0, 1707.0, 2859.0, 2108.0, 1359.0]
        L_f = [920.0, 3844.0, 2369.0, 1088.0, 3534.0, 1207.0, 17.0, 1041.0, 3512.0, 3418.0, 1188.0, 902.0, 2336.0, 3911.0, 1441.0, 141.0, 2690.0, 928.0, 39.0, 2762.0, 906.0, 838.0, 2657.0, 2125.0, 3565.0, 1967.0, 2291.0, 914.0, 932.0, 1620.0, 2160.0, 247.0, 222.0, 261.0, 2881.0, 2145.0, 3072.0, 1028.0, 1956.0, 2080.0, 1286.0, 3798.0, 1959.0, 28.0, 2248.0, 3247.0, 3594.0, 3155.0, 1345.0, 531.0, 1277.0, 593.0, 3044.0, 3083.0, 3005.0, 1380.0, 2020.0, 105.0, 1678.0, 1608.0, 2572.0, 3791.0, 1104.0, 2144.0, 318.0, 1186.0, 1073.0, 595.0, 2724.0, 1641.0, 351.0, 2908.0, 357.0, 3079.0, 1688.0, 3556.0, 3186.0, 2406.0, 224.0, 1962.0, 1480.0, 3251.0, 11.0, 345.0, 3526.0, 1784.0, 951.0, 3668.0, 2485.0, 1958.0, 2739.0, 916.0, 950.0, 2443.0, 3684.0, 904.0, 898.0, 587.0, 552.0, 2143.0, 3481.0, 3097.0, 3067.0, 1449.0, 47.0, 616.0, 3281.0, 1259.0, 661.0, 2348.0, 562.0, 3606.0, 2496.0, 2085.0, 1271.0, 372.0, 2857.0, 3325.0, 1394.0, 1081.0, 1032.0, 918.0, 1409.0, 314.0, 899.0, 733.0, 2245.0, 381.0, 2316.0, 232.0, 2405.0, 2677.0, 1066.0, 2396.0, 2282.0, 1059.0, 2622.0, 1941.0, 959.0, 3479.0, 3124.0, 1197.0, 1777.0, 915.0, 955.0, 1648.0, 3705.0, 3061.0, 34.0, 1285.0, 1.0, 2875.0, 1150.0, 3545.0, 2664.0, 2155.0, 1097.0, 262.0, 3915.0, 971.0, 2186.0, 3702.0, 3105.0, 2280.0, 3604.0, 3515.0, 1513.0, 2331.0, 1500.0, 2803.0, 945.0, 2639.0, 3051.0, 837.0, 3408.0, 457.0, 1801.0, 2506.0, 4.0, 2469.0, 270.0, 46.0, 1235.0, 2355.0, 2346.0, 1357.0, 461.0, 3255.0, 3176.0, 3350.0, 2975.0, 2014.0, 3936.0, 2072.0, 1353.0, 2006.0, 1397.0, 2612.0, 1099.0, 1367.0, 3270.0, 938.0, 2357.0, 94.0, 412.0, 1518.0, 3591.0, 538.0, 2000.0, 2846.0, 708.0, 329.0, 2995.0, 653.0, 1280.0, 5.0, 337.0, 1022.0, 2468.0, 1569.0, 905.0, 1031.0, 900.0, 1541.0, 2926.0, 3730.0, 1900.0, 2718.0, 1021.0, 3185.0, 2746.0, 327.0, 2805.0, 3101.0, 2920.0, 3269.0, 1674.0, 477.0, 3686.0, 2077.0, 2801.0, 581.0, 2133.0, 3724.0, 3296.0, 3554.0, 3478.0, 1479.0, 3720.0, 491.0, 1014.0, 1236.0, 3134.0, 695.0, 2763.0, 1013.0, 1096.0, 1856.0, 2827.0, 248.0, 1875.0, 3211.0, 3672.0, 215.0, 3224.0, 3396.0, 469.0, 1897.0, 3528.0, 2870.0, 917.0, 930.0, 1654.0, 3328.0, 3786.0, 907.0, 3870.0, 1422.0, 2206.0, 2114.0, 2324.0, 2575.0, 919.0, 3467.0, 1047.0, 1806.0, 350.0, 230.0, 2505.0, 48.0, 182.0, 144.0, 170.0, 2141.0, 1916.0, 3081.0, 1191.0, 1086.0, 2598.0, 546.0, 1407.0, 153.0, 2635.0, 2057.0, 2037.0, 1327.0, 3145.0, 446.0, 2193.0, 1337.0, 1913.0, 195.0, 2132.0, 1804.0, 3562.0, 3706.0, 1172.0, 1042.0, 2946.0, 2514.0, 1093.0, 1616.0, 3011.0, 2151.0, 1111.0, 613.0, 1043.0, 2774.0, 2154.0, 2621.0, 52.0, 3060.0, 3723.0, 206.0, 3133.0, 1821.0, 1964.0, 211.0, 2454.0, 532.0, 218.0, 3156.0, 1586.0, 1126.0, 2096.0, 927.0, 2007.0, 778.0, 2097.0, 3117.0, 691.0, 3567.0, 1223.0, 1268.0, 1300.0, 2747.0, 1573.0, 3302.0, 671.0, 3471.0, 3825.0, 1064.0, 1299.0, 252.0, 3004.0, 2091.0, 2337.0, 61.0, 1020.0, 3763.0, 1727.0, 74.0, 3599.0, 3708.0, 465.0, 29.0, 3741.0, 3457.0, 2399.0, 781.0, 69.0, 3635.0, 3808.0, 3249.0, 2732.0, 1621.0, 1686.0, 3435.0, 3857.0, 3299.0, 3426.0, 176.0, 343.0, 2972.0, 2853.0, 272.0, 2788.0, 1393.0, 203.0, 1465.0, 801.0, 1917.0, 2431.0, 3714.0, 2967.0, 3553.0, 79.0, 3951.0, 1683.0, 3071.0, 3102.0, 302.0, 3655.0, 2261.0, 3877.0, 2266.0, 3716.0, 3699.0, 1769.0, 266.0, 1173.0, 2693.0, 3093.0, 1658.0, 277.0, 279.0, 848.0, 839.0, 2365.0, 2738.0, 1264.0, 271.0, 1269.0, 2043.0, 3855.0, 1030.0, 1346.0, 2052.0, 2142.0, 2719.0, 2574.0, 2053.0, 1410.0, 3912.0, 1381.0, 3660.0, 2446.0, 2613.0, 2314.0, 978.0, 348.0, 2168.0, 3466.0, 669.0, 3649.0, 2448.0, 2899.0, 1611.0, 2940.0, 8.0, 1463.0, 26.0, 3557.0, 1994.0, 1758.0, 414.0, 1027.0, 3088.0, 3391.0, 1936.0, 2205.0, 3861.0, 332.0, 3450.0, 2585.0, 3618.0, 425.0, 1605.0, 3827.0, 846.0, 2267.0, 2359.0, 2952.0, 2786.0, 3923.0, 1290.0, 3240.0, 3388.0, 1547.0, 338.0, 3712.0, 3063.0, 242.0, 715.0, 3679.0, 3571.0, 668.0, 1069.0, 2276.0, 1438.0, 2688.0, 2900.0, 168.0, 3539.0, 199.0, 3675.0, 2436.0, 647.0, 724.0, 82.0, 542.0, 1362.0, 117.0, 2109.0, 3246.0, 3019.0, 1904.0, 360.0, 2966.0, 482.0, 2741.0, 334.0, 2100.0, 2173.0, 1615.0, 358.0, 280.0, 3932.0, 369.0, 3547.0, 3739.0, 1788.0, 875.0, 2106.0, 3719.0, 3839.0, 1417.0, 3566.0, 3795.0, 670.0, 520.0, 208.0, 3449.0, 3274.0, 27.0, 3872.0, 2969.0, 2927.0, 2442.0, 113.0, 2084.0, 1848.0, 3882.0, 3790.0, 3926.0, 2820.0, 3922.0, 3046.0, 832.0, 3896.0, 2101.0, 1600.0, 2548.0, 2453.0, 386.0, 239.0, 1015.0, 85.0, 3077.0, 3264.0, 3340.0, 3114.0, 1729.0, 2498.0, 309.0, 1034.0, 2421.0, 3438.0, 2599.0, 405.0, 3461.0, 3813.0, 3238.0, 3399.0, 3921.0, 912.0, 1840.0, 2876.0, 319.0, 40.0, 257.0, 3287.0, 880.0, 754.0, 1874.0, 2241.0, 2553.0, 1699.0, 550.0, 1549.0, 2338.0, 1922.0, 3612.0, 1894.0, 1049.0, 1185.0, 2779.0, 3902.0, 3580.0, 2.0, 2435.0, 73.0, 1012.0, 1275.0, 783.0, 512.0, 1919.0, 3838.0, 2903.0, 507.0, 1896.0, 2263.0, 2320.0, 1515.0, 363.0, 3492.0, 1562.0, 1588.0, 408.0, 3405.0, 307.0, 1199.0, 3268.0, 186.0, 1961.0, 1428.0, 2540.0, 3284.0, 2062.0, 3624.0, 1169.0, 2513.0, 575.0, 380.0, 2696.0, 2070.0, 2130.0, 3897.0, 615.0, 50.0, 3852.0, 415.0, 1797.0, 1660.0, 506.0, 3704.0, 2816.0, 2678.0, 2122.0, 1836.0, 2126.0, 481.0, 87.0, 3577.0, 2990.0, 3200.0, 441.0, 1554.0, 346.0, 1653.0, 2202.0, 2616.0, 283.0, 3584.0, 2417.0, 2284.0, 2042.0, 3454.0, 1582.0, 2568.0, 1669.0, 2048.0, 3613.0, 1911.0, 949.0, 420.0, 1719.0, 2361.0, 41.0, 3949.0, 379.0, 2379.0, 3447.0, 2136.0, 2642.0, 3206.0, 1995.0, 3150.0, 2856.0, 2010.0, 2532.0, 382.0, 2398.0, 1798.0, 1242.0, 2414.0, 2550.0, 1084.0, 131.0, 3055.0, 2630.0, 1949.0, 1954.0, 2352.0, 2110.0, 3181.0, 2021.0, 1344.0, 3685.0, 1398.0, 1312.0, 910.0, 3738.0, 173.0, 1456.0, 3445.0, 986.0, 2848.0, 2722.0, 3696.0, 3864.0, 3707.0, 1171.0, 558.0, 356.0, 2717.0, 3204.0, 2561.0, 934.0, 2704.0, 371.0, 1831.0, 879.0, 2439.0, 3108.0, 2517.0, 1372.0, 1672.0, 807.0, 3616.0, 688.0, 2797.0, 519.0, 1211.0, 1730.0, 1446.0, 1546.0, 2445.0, 2147.0, 3475.0, 1556.0, 1580.0, 1220.0, 2373.0, 501.0, 124.0, 1216.0, 1429.0, 2683.0, 2066.0, 1881.0, 2949.0, 3090.0, 802.0, 1870.0, 407.0, 586.0, 1944.0, 2989.0, 1921.0, 1226.0, 2380.0, 3489.0, 3886.0, 2190.0, 2919.0, 2495.0, 2392.0, 753.0, 1484.0, 1667.0, 2363.0, 3308.0, 1077.0, 1805.0, 2714.0, 3173.0, 216.0, 1694.0, 736.0, 1321.0, 1483.0, 608.0, 1485.0, 1347.0, 2789.0, 25.0, 2699.0, 1792.0, 2065.0, 2709.0, 2860.0, 1845.0, 2752.0, 494.0, 2273.0, 62.0, 2710.0, 866.0, 3841.0, 1566.0, 3153.0, 973.0, 3600.0, 1240.0, 1270.0, 923.0, 2159.0, 896.0, 3258.0, 147.0, 3439.0, 2947.0, 2643.0, 1212.0, 1258.0, 2527.0, 1419.0, 1217.0, 316.0, 1293.0, 2420.0, 3130.0, 2474.0, 2879.0, 991.0, 3317.0, 2713.0, 3440.0, 2463.0, 1619.0, 2539.0, 3070.0, 3040.0, 2163.0, 508.0, 428.0, 1816.0, 2533.0, 2736.0, 1969.0, 3054.0, 2176.0, 288.0, 2794.0, 2239.0, 2290.0, 1234.0, 3735.0, 2166.0, 19.0, 2071.0, 2394.0, 2858.0]
        # --- for this data need to find popular lm, lf
    elif dataset == 'yahoo':
        X = DL.load_user_item_matrix_yahoo()
        T = DL.load_gender_vector_yahoo()
        X_filled = DL.load_user_item_matrix_yahoo_Impute()
        # updated list with each users min rating 20
        L_m = [2587, 3581, 4289, 4049, 132, 916, 7038, 1111, 6790, 1691, 372, 5818, 7266, 1946, 3713, 7661, 2450, 6177, 1487, 4249, 6787, 6262, 4743, 6590, 7262, 8346, 7565, 5073, 5061, 5003, 1442, 7660, 1409, 7064, 2956, 7451, 3425, 1367, 5300, 5908, 7063, 2858, 3210, 292, 7288, 6750, 3123, 4507, 1278, 5373, 5040, 1134, 7895, 6763, 6539, 1483, 2802, 2998, 1066, 4016, 6547, 5164, 3471, 1430, 5532, 1556, 1106, 3239, 3887, 4217, 1415, 7558, 3582, 3534, 6574, 4343, 5729, 762, 6635, 4639, 802, 8568, 3948, 3724, 5577, 4789, 3326, 4481, 6185, 1165, 6811, 5592, 1615, 3755, 6376, 2590, 3258, 6582, 5582, 1376, 1799, 3199, 1555, 5227, 4358, 5265, 4522, 144, 6858, 8287, 1863, 6925, 6292, 6412, 6482, 4004, 5216, 7220, 7759, 2686, 2925, 5130, 2368, 177, 2366, 5013, 3249, 3245, 5937, 578, 2260, 984, 1351, 8141, 3940, 5555, 2115, 4459, 8315, 2693, 1867, 4252, 8136, 3153, 3186, 4056, 3487, 1947, 5935, 769, 1744, 6789, 5814, 4962, 6116, 2677, 8529, 4870, 3570, 6718, 4068, 2947, 1805, 5043, 6455, 6992, 6067, 2930, 3394, 6270, 4244, 7601, 8464, 2648, 5796, 6165, 2815, 5972, 6753, 6857, 3317, 3630, 327]
        L_f = [8569, 8176, 8494, 5099, 8218, 8533, 4931, 126, 760, 7813, 8563, 4468, 8219, 562, 8319, 4636, 1100, 8215, 8379, 1642, 8072, 8323, 3618, 7020, 7864, 7628, 4804, 441, 323, 719, 5302, 7885, 8390, 2315, 8306, 8238, 8301, 8253, 1160, 2405, 1970, 8177, 6944, 5675, 8093, 7656, 1576, 1362, 550, 4819, 6957, 939, 4234, 2258, 6970, 5448, 352, 7651, 7490, 8349, 7600, 54, 7781, 6221, 100, 7478, 92, 8430, 7081, 7587, 5039, 4233, 7592, 2972, 7498, 8506, 4903, 7778, 282, 8235, 6801, 6357, 8474, 303, 7972, 7630, 1621, 6948, 5984, 5391, 52, 36, 6991, 4464, 4893, 7883, 8039, 423, 7732, 3964, 291, 8531, 235, 5225, 1971, 1292, 4280, 5291, 7589, 210, 654, 361, 7557, 8459, 7834, 8134, 6932, 8227, 1101, 587, 7983, 4274, 606, 6967, 7005, 634, 7590, 110, 5841, 7860, 5521, 215, 4010, 542, 7996, 7466, 7990, 7644, 8418, 616, 8425, 8470, 8033, 388, 7756, 382, 5967, 7769, 4486, 5464, 7768, 384, 7705, 6761, 8370, 1908, 8092, 8318, 8398, 5825, 6937, 7772, 8362, 7703, 485, 7835, 8123, 5443, 2023, 8165, 5623, 7737, 5890, 8249, 2906, 4629, 8188, 149, 468, 407, 7987, 4892, 8003, 7964, 5376, 5687, 7655, 563, 6910, 4963, 7999, 7796, 8041, 4741, 4203, 4699, 8485, 6895, 5529, 6193, 7896, 597, 5159, 8027, 7479, 818, 7798, 6587, 601, 3807, 3, 8572, 6904, 2052, 621, 8266, 5850, 7483, 8048, 3941, 8486, 5404, 7936, 5134, 8303, 325, 3831, 8057, 6405, 8157, 373, 3013, 5621, 87, 6894, 8071, 5614, 5605, 8091, 8274, 8206, 329, 6488, 6837, 3826, 2323, 5025, 2494, 8058, 62, 3071, 6174, 5884, 5838, 7707, 5865, 8070, 356, 1250, 5539, 125, 7240, 7949, 3859, 182, 69, 4271, 8481, 8061, 5630, 7854, 3509, 7958, 10, 7502, 4525, 8083, 6462, 7873, 557, 154, 137, 5956, 7809, 8180, 6105, 3357, 5307, 5485, 1285, 6343, 7612, 8046, 7459, 6922, 70, 1713, 855, 8438, 204, 7945, 2085, 4337, 7747, 7633, 7941, 7012, 7843, 2567, 2522, 1872, 620, 558, 8226, 4851, 7973, 8233, 527, 7697, 213, 8214, 8421, 541, 8240, 4466, 6756, 7912, 8060, 5421, 5665, 2155, 3881, 7994, 3195, 7131, 8205, 2555, 6930, 7539, 4152, 7731, 3697, 7968, 607, 271, 5809, 239, 11, 4424, 3692, 8232, 7, 6239, 8295, 3073, 365, 5974, 1922, 1136, 2986, 7853, 2227, 7795, 6839, 1213, 8239, 524, 8124, 5579, 5530, 276, 519, 4813, 5552, 5034, 7704, 7788, 8162, 5308, 2234, 1416, 8472, 8225, 3292, 7842, 8154, 6999, 5342, 4514, 7634, 7613, 820, 8095, 3794, 6161, 102, 8327, 253, 7810, 7443, 1043, 107, 3949, 7741, 25, 254, 4981, 7519, 411, 3838, 7829, 8155, 8208, 5768, 8312, 3293, 6015, 7525, 8203, 71, 6997, 8244, 8326, 5755, 7579, 8343, 2017, 2386, 6849, 5052, 6091, 2583, 2088, 422, 8490, 4769, 469, 514, 6897, 575, 763, 3314, 3883, 8160, 8189, 5672, 7211, 5553, 6266, 8211, 7706, 8350, 805, 455, 5058, 4900, 403, 7775, 7876, 7461, 217, 6908, 8037, 7620, 7982, 1785, 7955, 8434, 8289, 7098, 6018, 5919, 5878, 5245, 8537, 8273, 538, 7838, 3614, 8212, 8193, 8210, 13, 8030, 5800, 842, 8250, 8269, 3580, 6211, 7520, 566, 8143, 8373, 6444, 3337, 8159, 4258, 7708, 8204, 7848, 7567, 151, 7937, 694, 473, 8190, 248, 8264]


    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = []
        for rating in X[:, item_id]:
            if rating > 0:
                ratings.append(rating)
        if len(ratings) == 0:
            avg_ratings[item_id] = 0
        else:
            avg_ratings[item_id] = np.average(ratings)
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor
    # 1: get the set of most correlated movies, L_f and L_m:
    with open('ml-'+dataset+'/NN_TrainingSet_AllUsers_Neighbors_Weight_K_30_item_choice_Top100IndicativeItems_noRemoval.json') as json_file:
        item_choice = json.load(json_file)

    # Now, where we have the two lists, we can start obfuscating the data:
    X_obf = np.copy(X)
    total_added = 0
    for index, user in enumerate(X):
        print(index)
        k = 0
        for rating in user:
            if rating > 0:
                k += 1
        k *= p
        greedy_index = 0
        added = 0
        mylist = list(item_choice.values())
        safety_counter = 0
        print(f"user: {index}, {user}, k = {k}")

        while added < k and safety_counter < top:
            if greedy_index >= len(mylist[index]):
                safety_counter = top
                continue
            if sample_mode == 'greedy':
                vec = mylist[index]
                movie_id = vec[greedy_index]
            if sample_mode == 'random':
                movie_id = vec[np.random.randint(0, len(vec))]
            greedy_index += 1
            rating_count = sum([1 if x > 0 else 0 for x in X_obf[:, movie_id]])
            if rating_count > max_count[movie_id]:
                continue

            if X_obf[index, movie_id] == 0:# and X_test [index, int(movie_id) ] == 0:

                X_obf[index, movie_id] =  avg_ratings[int(movie_id)] # X_filled[index, movie_id]
                added += 1
            safety_counter += 1
        total_added += added

    # Now remove ratings from users that have more than 20 ratings equally:
    if removal_mode == "strategic":
        nr_many_ratings = 0
        for user in X:
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                nr_many_ratings += 1
        print("nbr user with profile length > 20: ", nr_many_ratings)
        print("total_added: ", total_added)
        nr_remove = total_added / nr_many_ratings
        print("nr_remove: ", nr_remove)

        for user_index, user in enumerate(X):
            print("user: ", user_index)
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                index_m = 0
                index_f = 0
                rem = 0
                if T[user_index] == 1:
                    safety_counter = 0
                    # We note that if we add safety_counter < 1000 in the while we have a higher accuracy than if we keep it in the if
                    while (rem < nr_remove) and safety_counter < top:
                        if index_f >= len(L_f):  # and safety_counter < 1000:
                            safety_counter = top
                            continue

                        to_be_removed_indecies = L_f[index_f]
                        index_f += 1

                        if X_obf[user_index, int(to_be_removed_indecies)-1] != 0:
                            X_obf[user_index, int(to_be_removed_indecies)-1 ] = 0
                            rem += 1
                        safety_counter += 1

                elif T[user_index] == 0:

                    while (rem < nr_remove) and safety_counter < top:
                        if index_m >= len(L_m):  # and safety_counter < 1000:
                            safety_counter = top
                            continue

                        to_be_removed_indecies = L_m[index_m]
                        index_m += 1

                        if X_obf[user_index, int(to_be_removed_indecies) - 1] != 0:
                            X_obf[user_index, int(to_be_removed_indecies) - 1] = 0
                            rem += 1
                        safety_counter += 1
    else:
        # Now remove ratings from users that have more than 200 ratings equally:

        nr_many_ratings = 0
        for user in X:
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                nr_many_ratings += 1
        nr_remove = total_added / nr_many_ratings

        for user_index, user in enumerate(X):
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                to_be_removed_indecies = np.random.choice(np.argwhere(user > 0)[:, 0], size=(int(nr_remove),)) #  replace=False
                X_obf[user_index, to_be_removed_indecies] = 0


    # Save the obfuscated data to a file
    output_file = 'ml-'+dataset+'/PerBlur/'
    print(output_file)
    with open(output_file + "PerBlurwithRemoval_" + rating_mode + "_" + sample_mode + "_" + str(p) + ".dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(int(np.round(rating))) + "::000000000\n")


    return X_obf

# --- Proposed Approach --- #

def SmartBlur():

    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    p = 0.1
    notice_factor = 2
    dataset = ['100k', '1m', 'yahoo'][2]

    if dataset == '100k':
        X = DL.load_user_item_matrix_100k()
        T = DL.load_gender_vector_100k()
        X_filled = DL.load_user_item_matrix_100k_Impute()
        L_m = [751, 327, 24, 688, 472, 888, 317, 948, 315, 178, 305, 179, 359, 186, 10, 883, 294, 342, 5, 471, 1048, 1101, 879, 449, 324, 150, 357, 411, 1265, 136, 524, 678, 202, 692, 100, 303, 264, 339, 1010, 276, 218, 995, 240, 433, 271, 203, 544, 520, 410, 222, 1028, 924, 823, 322, 462, 265, 750, 23, 281, 137, 333, 736, 164, 358, 301, 479, 647, 168, 144, 94, 687, 60, 70, 175, 6, 354, 344, 1115, 597, 596, 42, 14, 521, 184, 393, 188, 450, 1047, 682, 7, 760, 737, 270, 1008, 260, 295, 1051, 289, 430, 235, 258, 330, 56, 347, 52, 205, 371, 1024, 8, 369, 116, 79, 831, 694, 287, 435, 495, 242, 33, 201, 230, 506, 979, 620, 355, 181, 717, 936, 500, 893, 489, 1316, 855, 2, 180, 871, 755, 523, 477, 227, 412, 183, 657, 510, 77, 474, 1105, 67, 705, 362, 1133, 511, 546, 768, 96, 457, 108, 1137, 352, 341, 825, 165, 584, 505, 908, 404, 679, 15, 513, 117, 490, 665, 28, 338, 59, 32, 1014, 989, 351, 475, 57, 864, 969, 177, 316, 463, 134, 703, 306, 378, 105, 99, 229, 484, 651, 157, 232, 114, 161, 395, 754, 931, 591, 408, 685, 97, 554, 468, 455, 640, 473, 683, 300, 31, 1060, 650, 72, 191, 259, 1280, 199, 826, 747, 68, 1079, 887, 578, 329, 274, 174, 428, 905, 780, 753, 396, 4, 277, 71, 519, 1062, 189, 325, 502, 233, 1022, 880, 1063, 197, 813, 16, 331, 208, 162, 11, 963, 501, 820, 930, 896, 318, 1142, 1194, 425, 171, 282, 496, 26, 215, 573, 527, 730, 49, 693, 517, 336, 417, 207, 900, 299, 226, 606, 268, 192, 407, 62, 636, 480, 1016, 241, 945, 343, 603, 231, 469, 515, 492, 664, 972, 642, 781, 984, 187, 149, 257, 40, 141, 978, 44, 845, 85, 244, 512, 182, 1021, 756, 1, 778, 644, 1050, 185, 530, 81, 497, 3, 335, 923, 509, 418, 724, 566, 221, 570, 655, 413, 1311, 340, 583, 537, 635, 686, 176, 296, 926, 561, 101, 173, 862, 680, 652]
        L_f = [332, 321, 877, 906, 827, 882, 337, 278, 872, 304, 87, 93, 1038, 292, 629, 727, 220, 898, 107, 143, 516, 819, 382, 499, 876, 45, 1600, 111, 416, 132, 752, 125, 689, 604, 628, 310, 246, 419, 148, 659, 881, 248, 899, 707, 20, 129, 272, 690, 350, 498, 19, 216, 713, 429, 253, 372, 311, 251, 288, 1023, 392, 1328, 131, 320, 83, 895, 912, 445, 1094, 432, 109, 156, 532, 588, 486, 653, 243, 319, 518, 269, 298, 447, 290, 1386, 988, 423, 723, 279, 696, 568, 919, 1296, 266, 645, 937, 595, 482, 748, 348, 749, 815, 451, 154, 684, 993, 904, 204, 198, 403, 525, 553, 671, 507, 1119, 1036, 65, 420, 169, 9, 427, 832, 894, 729, 654, 1128, 987, 1281, 401, 1089, 942, 190, 406, 170, 402, 878, 291, 710, 297, 721, 151, 367, 954, 856, 64, 312, 385, 581, 370, 39, 648, 691, 746, 89, 1033, 745, 194, 662, 145, 539, 98, 147, 739, 159, 90, 236, 55, 120, 448, 200, 409, 623, 118, 731, 365, 238, 124, 146, 716, 1026, 847, 261, 153, 262, 130, 127, 866, 58, 356, 195, 1086, 1190, 1041, 1073, 889, 405, 950, 891, 611, 582, 66, 1035, 346, 1394, 172, 909, 1013, 785, 925, 488, 704, 1315, 708, 1431, 309, 121, 239, 902, 213, 446, 1012, 1054, 873, 193, 478, 829, 916, 353, 349, 133, 249, 550, 638, 452, 1313, 466, 529, 458, 792, 313, 625, 668, 1061, 742, 1065, 763, 73, 846, 1090, 863, 1620, 890, 1176, 1017, 237, 1294, 245, 508, 387, 53, 514, 422, 1068, 1527, 939, 1232, 1011, 631, 381, 956, 875, 459, 607, 1442, 155, 697, 1066, 1285, 293, 88, 1221, 1109, 675, 254, 209, 649, 140, 48, 613, 962, 1157, 991, 1083, 720, 535, 901, 436, 286, 559, 1085, 892, 102, 211, 285, 1383, 758, 1234, 674, 1163, 283, 637, 735, 885, 630, 709, 938, 38, 142, 219, 953, 275, 1059, 676, 210, 47, 63, 255, 494, 744, 250, 1114, 672, 308, 632, 17, 1203, 762, 1522, 538, 491, 307, 669, 775, 1135, 217, 841, 949, 766, 966, 314, 812, 821, 574, 1039, 1388, 51, 579, 252, 50, 1009, 1040, 1147, 224, 212, 1483, 1278, 122, 934, 702, 443, 86, 1444, 69, 400, 975, 35, 549, 928, 531, 622, 614, 1020, 658, 740, 1468, 22, 1007, 470, 1269, 633, 1074, 1299, 1095, 1148, 280, 1136, 92, 167, 434, 284, 961, 1084, 126, 619, 974, 196, 485, 1152, 673, 627, 345, 29, 605, 929, 952, 714, 1053, 526, 123, 476, 660, 955, 380, 1503, 163, 493, 1197, 431, 504, 886, 1037, 13, 794, 273, 844, 1032, 1025, 106, 91, 533, 421, 699, 869, 78, 1243, 481, 661, 82, 732]

    elif dataset == '1m':
        X = DL.load_user_item_matrix_1m()
        T = DL.load_gender_vector_1m()
        X_filled = DL.load_user_item_matrix_1m_Impute()
        L_m = [589.0, 1233.0, 2706.0, 1090.0, 2959.0, 1250.0, 2427.0, 2490.0, 1208.0, 1266.0, 3654.0, 1748.0, 1262.0, 1396.0, 1374.0, 2700.0, 1265.0, 1089.0, 1222.0, 231.0, 2770.0, 1676.0, 2890.0, 1228.0, 1136.0, 3360.0, 3298.0, 1663.0, 3811.0, 2011.0, 1261.0, 233.0, 3361.0, 2366.0, 1127.0, 1276.0, 3555.0, 1214.0, 3929.0, 299.0, 1304.0, 3468.0, 1095.0, 150.0, 1213.0, 750.0, 3082.0, 6.0, 111.0, 3745.0, 349.0, 541.0, 2791.0, 785.0, 1060.0, 1294.0, 1302.0, 1256.0, 1292.0, 2948.0, 3683.0, 3030.0, 3836.0, 913.0, 2150.0, 32.0, 2826.0, 2721.0, 590.0, 3623.0, 2997.0, 3868.0, 3147.0, 1610.0, 3508.0, 2046.0, 21.0, 1249.0, 10.0, 1283.0, 3760.0, 2712.0, 3617.0, 3552.0, 3256.0, 1079.0, 3053.0, 1517.0, 2662.0, 1953.0, 2670.0, 3578.0, 2371.0, 3334.0, 2502.0, 2278.0, 364.0, 3462.0, 2401.0, 3163.0, 2311.0, 852.0, 2916.0, 1378.0, 3384.0, 524.0, 70.0, 370.0, 3035.0, 3513.0, 2917.0, 3697.0, 24.0, 1957.0, 3494.0, 1912.0, 3752.0, 2013.0, 3452.0, 3928.0, 2987.0, 431.0, 2759.0, 1387.0, 1882.0, 3638.0, 1288.0, 2867.0, 2728.0, 2433.0, 161.0, 3386.0, 517.0, 741.0, 1287.0, 1231.0, 3062.0, 2288.0, 3753.0, 529.0, 3793.0, 3052.0, 2447.0, 1320.0, 3819.0, 1303.0, 922.0, 3022.0, 260.0, 858.0, 493.0, 3006.0, 480.0, 2410.0, 333.0, 1178.0, 3814.0, 2702.0, 1203.0, 2922.0, 1625.0, 3366.0, 3213.0, 2188.0, 2628.0, 3358.0, 2648.0, 3788.0, 953.0, 999.0, 3754.0, 3910.0, 3016.0, 3863.0, 303.0, 3263.0, 1080.0, 786.0, 3764.0, 2105.0, 3543.0, 2607.0, 3681.0, 592.0, 145.0, 2303.0, 1682.0, 1019.0, 3646.0, 1544.0, 235.0, 908.0, 3615.0, 2792.0, 2167.0, 2455.0, 1587.0, 1227.0, 2901.0, 2687.0, 1883.0, 1210.0, 1201.0, 3169.0, 3098.0, 3688.0, 2409.0, 3198.0, 610.0, 1923.0, 1982.0, 165.0, 2403.0, 784.0, 2871.0, 2889.0, 628.0, 2300.0, 417.0, 3671.0, 3100.0, 3914.0, 3608.0, 3152.0, 3429.0, 1794.0, 952.0, 1391.0, 2518.0, 410.0, 3535.0, 2333.0, 1713.0, 2605.0, 707.0, 2795.0, 1965.0, 373.0, 3916.0, 556.0, 3703.0, 95.0, 466.0, 3066.0, 3177.0, 2088.0, 1476.0, 163.0, 3422.0, 58.0, 1244.0, 1689.0, 2002.0, 1711.0, 2259.0, 3524.0, 1371.0, 3104.0, 1693.0, 965.0, 1732.0, 2600.0, 3424.0, 3755.0, 2450.0, 3826.0, 3801.0, 3927.0, 1298.0, 2118.0, 112.0, 2478.0, 471.0, 1673.0, 1246.0, 2734.0, 2529.0, 2806.0, 1948.0, 2093.0, 45.0, 648.0, 3504.0, 2968.0, 1722.0, 1963.0, 2840.0, 1747.0, 1348.0, 3871.0, 3175.0, 2360.0, 1092.0, 3190.0, 1405.0, 367.0, 3248.0, 1702.0, 1734.0, 2644.0, 1597.0, 1401.0, 1416.0, 107.0, 1379.0, 2764.0, 2116.0, 1036.0, 60.0, 2115.0, 1876.0, 1254.0, 2243.0, 2606.0, 3925.0, 3087.0, 1627.0, 3770.0, 3678.0, 3113.0, 3036.0, 3525.0, 1584.0, 2236.0, 3267.0, 954.0, 1205.0, 2470.0, 2686.0, 3397.0, 2015.0, 1377.0, 3740.0, 1594.0, 2456.0, 2038.0, 891.0, 1342.0, 1966.0, 2808.0, 3324.0, 3794.0, 2467.0, 3420.0, 3773.0, 1927.0, 2231.0, 3742.0, 1960.0, 1542.0, 2672.0, 1376.0, 3174.0, 1248.0, 225.0, 1267.0, 3203.0, 1025.0, 2769.0, 1973.0, 2541.0, 3593.0, 2058.0, 3273.0, 154.0, 1179.0, 2009.0, 2423.0, 2676.0, 2793.0, 3505.0, 1920.0, 3357.0, 2580.0, 2542.0, 1701.0, 3252.0, 440.0, 540.0, 1885.0, 2384.0, 1414.0, 1251.0, 1187.0, 2841.0, 2287.0, 2004.0, 1257.0, 1358.0, 2253.0, 3918.0, 2976.0, 1100.0, 2140.0, 2092.0, 2772.0, 3500.0, 1196.0, 3728.0, 555.0, 3564.0, 3099.0, 2863.0, 2492.0, 13.0, 2378.0, 3271.0, 3946.0, 1017.0, 3189.0, 3908.0, 1238.0, 3551.0, 800.0, 1193.0, 3254.0, 3614.0, 448.0, 1779.0, 3477.0, 1388.0, 748.0, 1411.0, 3948.0, 1057.0, 2877.0, 2633.0, 3078.0, 2289.0, 514.0, 3831.0, 535.0, 361.0, 290.0, 1408.0, 1356.0, 2522.0, 2321.0, 1395.0, 1103.0, 2861.0, 1974.0, 2497.0, 1633.0, 2530.0, 1931.0, 125.0, 1735.0, 3159.0, 892.0, 2828.0, 523.0, 3148.0, 296.0, 2882.0, 1639.0, 1665.0, 3834.0, 534.0, 2942.0, 1247.0, 861.0, 2107.0, 3469.0, 1970.0, 3307.0, 432.0, 3879.0, 3930.0, 742.0, 3937.0, 1237.0, 1091.0, 3214.0, 1273.0, 3809.0, 3115.0, 2111.0, 468.0, 3769.0, 2961.0, 3771.0, 246.0, 3094.0, 2907.0, 1016.0, 151.0, 377.0, 450.0, 3538.0, 3717.0, 2694.0, 2745.0, 2389.0, 3865.0, 281.0, 2272.0, 2991.0, 1810.0, 2024.0, 2725.0, 2731.0, 409.0, 2971.0, 1083.0, 2701.0, 1753.0, 1459.0, 2567.0, 673.0, 3516.0, 611.0, 947.0, 1176.0, 1640.0, 172.0, 2671.0, 2041.0, 2723.0, 2471.0, 378.0, 3901.0, 1834.0, 1733.0, 1135.0, 998.0, 2475.0, 292.0, 3347.0, 2121.0, 3952.0, 1219.0, 413.0, 2294.0, 1997.0, 849.0, 2017.0, 2025.0, 3476.0, 1399.0, 2822.0, 2068.0, 180.0, 2076.0, 3700.0, 1783.0, 3326.0, 1760.0, 2437.0, 3893.0, 2594.0, 16.0, 1942.0, 2171.0, 2815.0, 1281.0, 1589.0, 936.0, 3168.0, 2520.0, 3095.0, 3448.0, 1971.0, 1230.0, 3129.0, 3799.0, 3125.0, 3784.0, 3789.0, 3262.0, 1946.0, 2390.0, 1918.0, 3201.0, 3909.0, 2943.0, 2082.0, 3157.0, 2112.0, 3409.0, 1772.0, 1680.0, 3633.0, 2153.0, 720.0, 674.0, 3713.0, 126.0, 585.0, 2353.0, 158.0, 3676.0, 3398.0, 485.0, 765.0, 1284.0, 2089.0, 1148.0, 1147.0, 2183.0, 1037.0, 2393.0, 2250.0, 2524.0, 1617.0, 1457.0, 3135.0, 3142.0, 2935.0, 1461.0, 533.0, 1425.0, 1282.0, 728.0, 3521.0, 1972.0, 1361.0, 551.0, 2016.0, 454.0, 3889.0, 3837.0, 190.0, 2735.0, 2124.0, 2310.0, 23.0, 3548.0, 1466.0, 3743.0, 1124.0, 2033.0, 1590.0, 2138.0, 2716.0, 1649.0, 1189.0, 2135.0, 3243.0, 3359.0, 1339.0, 123.0, 1224.0, 2996.0, 344.0, 1101.0, 515.0, 2428.0, 1873.0, 1392.0, 2583.0, 258.0, 2519.0, 2771.0, 213.0, 451.0, 2906.0, 2313.0, 3253.0, 1343.0, 2941.0, 745.0, 2729.0, 353.0, 1707.0, 2859.0, 2108.0, 1359.0]
        L_f = [920.0, 3844.0, 2369.0, 1088.0, 3534.0, 1207.0, 17.0, 1041.0, 3512.0, 3418.0, 1188.0, 902.0, 2336.0, 3911.0, 1441.0, 141.0, 2690.0, 928.0, 39.0, 2762.0, 906.0, 838.0, 2657.0, 2125.0, 3565.0, 1967.0, 2291.0, 914.0, 932.0, 1620.0, 2160.0, 247.0, 222.0, 261.0, 2881.0, 2145.0, 3072.0, 1028.0, 1956.0, 2080.0, 1286.0, 3798.0, 1959.0, 28.0, 2248.0, 3247.0, 3594.0, 3155.0, 1345.0, 531.0, 1277.0, 593.0, 3044.0, 3083.0, 3005.0, 1380.0, 2020.0, 105.0, 1678.0, 1608.0, 2572.0, 3791.0, 1104.0, 2144.0, 318.0, 1186.0, 1073.0, 595.0, 2724.0, 1641.0, 351.0, 2908.0, 357.0, 3079.0, 1688.0, 3556.0, 3186.0, 2406.0, 224.0, 1962.0, 1480.0, 3251.0, 11.0, 345.0, 3526.0, 1784.0, 951.0, 3668.0, 2485.0, 1958.0, 2739.0, 916.0, 950.0, 2443.0, 3684.0, 904.0, 898.0, 587.0, 552.0, 2143.0, 3481.0, 3097.0, 3067.0, 1449.0, 47.0, 616.0, 3281.0, 1259.0, 661.0, 2348.0, 562.0, 3606.0, 2496.0, 2085.0, 1271.0, 372.0, 2857.0, 3325.0, 1394.0, 1081.0, 1032.0, 918.0, 1409.0, 314.0, 899.0, 733.0, 2245.0, 381.0, 2316.0, 232.0, 2405.0, 2677.0, 1066.0, 2396.0, 2282.0, 1059.0, 2622.0, 1941.0, 959.0, 3479.0, 3124.0, 1197.0, 1777.0, 915.0, 955.0, 1648.0, 3705.0, 3061.0, 34.0, 1285.0, 1.0, 2875.0, 1150.0, 3545.0, 2664.0, 2155.0, 1097.0, 262.0, 3915.0, 971.0, 2186.0, 3702.0, 3105.0, 2280.0, 3604.0, 3515.0, 1513.0, 2331.0, 1500.0, 2803.0, 945.0, 2639.0, 3051.0, 837.0, 3408.0, 457.0, 1801.0, 2506.0, 4.0, 2469.0, 270.0, 46.0, 1235.0, 2355.0, 2346.0, 1357.0, 461.0, 3255.0, 3176.0, 3350.0, 2975.0, 2014.0, 3936.0, 2072.0, 1353.0, 2006.0, 1397.0, 2612.0, 1099.0, 1367.0, 3270.0, 938.0, 2357.0, 94.0, 412.0, 1518.0, 3591.0, 538.0, 2000.0, 2846.0, 708.0, 329.0, 2995.0, 653.0, 1280.0, 5.0, 337.0, 1022.0, 2468.0, 1569.0, 905.0, 1031.0, 900.0, 1541.0, 2926.0, 3730.0, 1900.0, 2718.0, 1021.0, 3185.0, 2746.0, 327.0, 2805.0, 3101.0, 2920.0, 3269.0, 1674.0, 477.0, 3686.0, 2077.0, 2801.0, 581.0, 2133.0, 3724.0, 3296.0, 3554.0, 3478.0, 1479.0, 3720.0, 491.0, 1014.0, 1236.0, 3134.0, 695.0, 2763.0, 1013.0, 1096.0, 1856.0, 2827.0, 248.0, 1875.0, 3211.0, 3672.0, 215.0, 3224.0, 3396.0, 469.0, 1897.0, 3528.0, 2870.0, 917.0, 930.0, 1654.0, 3328.0, 3786.0, 907.0, 3870.0, 1422.0, 2206.0, 2114.0, 2324.0, 2575.0, 919.0, 3467.0, 1047.0, 1806.0, 350.0, 230.0, 2505.0, 48.0, 182.0, 144.0, 170.0, 2141.0, 1916.0, 3081.0, 1191.0, 1086.0, 2598.0, 546.0, 1407.0, 153.0, 2635.0, 2057.0, 2037.0, 1327.0, 3145.0, 446.0, 2193.0, 1337.0, 1913.0, 195.0, 2132.0, 1804.0, 3562.0, 3706.0, 1172.0, 1042.0, 2946.0, 2514.0, 1093.0, 1616.0, 3011.0, 2151.0, 1111.0, 613.0, 1043.0, 2774.0, 2154.0, 2621.0, 52.0, 3060.0, 3723.0, 206.0, 3133.0, 1821.0, 1964.0, 211.0, 2454.0, 532.0, 218.0, 3156.0, 1586.0, 1126.0, 2096.0, 927.0, 2007.0, 778.0, 2097.0, 3117.0, 691.0, 3567.0, 1223.0, 1268.0, 1300.0, 2747.0, 1573.0, 3302.0, 671.0, 3471.0, 3825.0, 1064.0, 1299.0, 252.0, 3004.0, 2091.0, 2337.0, 61.0, 1020.0, 3763.0, 1727.0, 74.0, 3599.0, 3708.0, 465.0, 29.0, 3741.0, 3457.0, 2399.0, 781.0, 69.0, 3635.0, 3808.0, 3249.0, 2732.0, 1621.0, 1686.0, 3435.0, 3857.0, 3299.0, 3426.0, 176.0, 343.0, 2972.0, 2853.0, 272.0, 2788.0, 1393.0, 203.0, 1465.0, 801.0, 1917.0, 2431.0, 3714.0, 2967.0, 3553.0, 79.0, 3951.0, 1683.0, 3071.0, 3102.0, 302.0, 3655.0, 2261.0, 3877.0, 2266.0, 3716.0, 3699.0, 1769.0, 266.0, 1173.0, 2693.0, 3093.0, 1658.0, 277.0, 279.0, 848.0, 839.0, 2365.0, 2738.0, 1264.0, 271.0, 1269.0, 2043.0, 3855.0, 1030.0, 1346.0, 2052.0, 2142.0, 2719.0, 2574.0, 2053.0, 1410.0, 3912.0, 1381.0, 3660.0, 2446.0, 2613.0, 2314.0, 978.0, 348.0, 2168.0, 3466.0, 669.0, 3649.0, 2448.0, 2899.0, 1611.0, 2940.0, 8.0, 1463.0, 26.0, 3557.0, 1994.0, 1758.0, 414.0, 1027.0, 3088.0, 3391.0, 1936.0, 2205.0, 3861.0, 332.0, 3450.0, 2585.0, 3618.0, 425.0, 1605.0, 3827.0, 846.0, 2267.0, 2359.0, 2952.0, 2786.0, 3923.0, 1290.0, 3240.0, 3388.0, 1547.0, 338.0, 3712.0, 3063.0, 242.0, 715.0, 3679.0, 3571.0, 668.0, 1069.0, 2276.0, 1438.0, 2688.0, 2900.0, 168.0, 3539.0, 199.0, 3675.0, 2436.0, 647.0, 724.0, 82.0, 542.0, 1362.0, 117.0, 2109.0, 3246.0, 3019.0, 1904.0, 360.0, 2966.0, 482.0, 2741.0, 334.0, 2100.0, 2173.0, 1615.0, 358.0, 280.0, 3932.0, 369.0, 3547.0, 3739.0, 1788.0, 875.0, 2106.0, 3719.0, 3839.0, 1417.0, 3566.0, 3795.0, 670.0, 520.0, 208.0, 3449.0, 3274.0, 27.0, 3872.0, 2969.0, 2927.0, 2442.0, 113.0, 2084.0, 1848.0, 3882.0, 3790.0, 3926.0, 2820.0, 3922.0, 3046.0, 832.0, 3896.0, 2101.0, 1600.0, 2548.0, 2453.0, 386.0, 239.0, 1015.0, 85.0, 3077.0, 3264.0, 3340.0, 3114.0, 1729.0, 2498.0, 309.0, 1034.0, 2421.0, 3438.0, 2599.0, 405.0, 3461.0, 3813.0, 3238.0, 3399.0, 3921.0, 912.0, 1840.0, 2876.0, 319.0, 40.0, 257.0, 3287.0, 880.0, 754.0, 1874.0, 2241.0, 2553.0, 1699.0, 550.0, 1549.0, 2338.0, 1922.0, 3612.0, 1894.0, 1049.0, 1185.0, 2779.0, 3902.0, 3580.0, 2.0, 2435.0, 73.0, 1012.0, 1275.0, 783.0, 512.0, 1919.0, 3838.0, 2903.0, 507.0, 1896.0, 2263.0, 2320.0, 1515.0, 363.0, 3492.0, 1562.0, 1588.0, 408.0, 3405.0, 307.0, 1199.0, 3268.0, 186.0, 1961.0, 1428.0, 2540.0, 3284.0, 2062.0, 3624.0, 1169.0, 2513.0, 575.0, 380.0, 2696.0, 2070.0, 2130.0, 3897.0, 615.0, 50.0, 3852.0, 415.0, 1797.0, 1660.0, 506.0, 3704.0, 2816.0, 2678.0, 2122.0, 1836.0, 2126.0, 481.0, 87.0, 3577.0, 2990.0, 3200.0, 441.0, 1554.0, 346.0, 1653.0, 2202.0, 2616.0, 283.0, 3584.0, 2417.0, 2284.0, 2042.0, 3454.0, 1582.0, 2568.0, 1669.0, 2048.0, 3613.0, 1911.0, 949.0, 420.0, 1719.0, 2361.0, 41.0, 3949.0, 379.0, 2379.0, 3447.0, 2136.0, 2642.0, 3206.0, 1995.0, 3150.0, 2856.0, 2010.0, 2532.0, 382.0, 2398.0, 1798.0, 1242.0, 2414.0, 2550.0, 1084.0, 131.0, 3055.0, 2630.0, 1949.0, 1954.0, 2352.0, 2110.0, 3181.0, 2021.0, 1344.0, 3685.0, 1398.0, 1312.0, 910.0, 3738.0, 173.0, 1456.0, 3445.0, 986.0, 2848.0, 2722.0, 3696.0, 3864.0, 3707.0, 1171.0, 558.0, 356.0, 2717.0, 3204.0, 2561.0, 934.0, 2704.0, 371.0, 1831.0, 879.0, 2439.0, 3108.0, 2517.0, 1372.0, 1672.0, 807.0, 3616.0, 688.0, 2797.0, 519.0, 1211.0, 1730.0, 1446.0, 1546.0, 2445.0, 2147.0, 3475.0, 1556.0, 1580.0, 1220.0, 2373.0, 501.0, 124.0, 1216.0, 1429.0, 2683.0, 2066.0, 1881.0, 2949.0, 3090.0, 802.0, 1870.0, 407.0, 586.0, 1944.0, 2989.0, 1921.0, 1226.0, 2380.0, 3489.0, 3886.0, 2190.0, 2919.0, 2495.0, 2392.0, 753.0, 1484.0, 1667.0, 2363.0, 3308.0, 1077.0, 1805.0, 2714.0, 3173.0, 216.0, 1694.0, 736.0, 1321.0, 1483.0, 608.0, 1485.0, 1347.0, 2789.0, 25.0, 2699.0, 1792.0, 2065.0, 2709.0, 2860.0, 1845.0, 2752.0, 494.0, 2273.0, 62.0, 2710.0, 866.0, 3841.0, 1566.0, 3153.0, 973.0, 3600.0, 1240.0, 1270.0, 923.0, 2159.0, 896.0, 3258.0, 147.0, 3439.0, 2947.0, 2643.0, 1212.0, 1258.0, 2527.0, 1419.0, 1217.0, 316.0, 1293.0, 2420.0, 3130.0, 2474.0, 2879.0, 991.0, 3317.0, 2713.0, 3440.0, 2463.0, 1619.0, 2539.0, 3070.0, 3040.0, 2163.0, 508.0, 428.0, 1816.0, 2533.0, 2736.0, 1969.0, 3054.0, 2176.0, 288.0, 2794.0, 2239.0, 2290.0, 1234.0, 3735.0, 2166.0, 19.0, 2071.0, 2394.0, 2858.0]
        # --- for this data need to find popular lm, lf
    elif dataset == 'yahoo':
        X = DL.load_user_item_matrix_yahoo()
        T = DL.load_gender_vector_yahoo()
        X_filled = DL.load_user_item_matrix_yahoo_Impute()
        # updated list with each users min rating 20
        L_m = [2587, 3581, 4289, 4049, 132, 916, 7038, 1111, 6790, 1691, 372, 5818, 7266, 1946, 3713, 7661, 2450, 6177, 1487, 4249, 6787, 6262, 4743, 6590, 7262, 8346, 7565, 5073, 5061, 5003, 1442, 7660, 1409, 7064, 2956, 7451, 3425, 1367, 5300, 5908, 7063, 2858, 3210, 292, 7288, 6750, 3123, 4507, 1278, 5373, 5040, 1134, 7895, 6763, 6539, 1483, 2802, 2998, 1066, 4016, 6547, 5164, 3471, 1430, 5532, 1556, 1106, 3239, 3887, 4217, 1415, 7558, 3582, 3534, 6574, 4343, 5729, 762, 6635, 4639, 802, 8568, 3948, 3724, 5577, 4789, 3326, 4481, 6185, 1165, 6811, 5592, 1615, 3755, 6376, 2590, 3258, 6582, 5582, 1376, 1799, 3199, 1555, 5227, 4358, 5265, 4522, 144, 6858, 8287, 1863, 6925, 6292, 6412, 6482, 4004, 5216, 7220, 7759, 2686, 2925, 5130, 2368, 177, 2366, 5013, 3249, 3245, 5937, 578, 2260, 984, 1351, 8141, 3940, 5555, 2115, 4459, 8315, 2693, 1867, 4252, 8136, 3153, 3186, 4056, 3487, 1947, 5935, 769, 1744, 6789, 5814, 4962, 6116, 2677, 8529, 4870, 3570, 6718, 4068, 2947, 1805, 5043, 6455, 6992, 6067, 2930, 3394, 6270, 4244, 7601, 8464, 2648, 5796, 6165, 2815, 5972, 6753, 6857, 3317, 3630, 327]
        L_f = [8569, 8176, 8494, 5099, 8218, 8533, 4931, 126, 760, 7813, 8563, 4468, 8219, 562, 8319, 4636, 1100, 8215, 8379, 1642, 8072, 8323, 3618, 7020, 7864, 7628, 4804, 441, 323, 719, 5302, 7885, 8390, 2315, 8306, 8238, 8301, 8253, 1160, 2405, 1970, 8177, 6944, 5675, 8093, 7656, 1576, 1362, 550, 4819, 6957, 939, 4234, 2258, 6970, 5448, 352, 7651, 7490, 8349, 7600, 54, 7781, 6221, 100, 7478, 92, 8430, 7081, 7587, 5039, 4233, 7592, 2972, 7498, 8506, 4903, 7778, 282, 8235, 6801, 6357, 8474, 303, 7972, 7630, 1621, 6948, 5984, 5391, 52, 36, 6991, 4464, 4893, 7883, 8039, 423, 7732, 3964, 291, 8531, 235, 5225, 1971, 1292, 4280, 5291, 7589, 210, 654, 361, 7557, 8459, 7834, 8134, 6932, 8227, 1101, 587, 7983, 4274, 606, 6967, 7005, 634, 7590, 110, 5841, 7860, 5521, 215, 4010, 542, 7996, 7466, 7990, 7644, 8418, 616, 8425, 8470, 8033, 388, 7756, 382, 5967, 7769, 4486, 5464, 7768, 384, 7705, 6761, 8370, 1908, 8092, 8318, 8398, 5825, 6937, 7772, 8362, 7703, 485, 7835, 8123, 5443, 2023, 8165, 5623, 7737, 5890, 8249, 2906, 4629, 8188, 149, 468, 407, 7987, 4892, 8003, 7964, 5376, 5687, 7655, 563, 6910, 4963, 7999, 7796, 8041, 4741, 4203, 4699, 8485, 6895, 5529, 6193, 7896, 597, 5159, 8027, 7479, 818, 7798, 6587, 601, 3807, 3, 8572, 6904, 2052, 621, 8266, 5850, 7483, 8048, 3941, 8486, 5404, 7936, 5134, 8303, 325, 3831, 8057, 6405, 8157, 373, 3013, 5621, 87, 6894, 8071, 5614, 5605, 8091, 8274, 8206, 329, 6488, 6837, 3826, 2323, 5025, 2494, 8058, 62, 3071, 6174, 5884, 5838, 7707, 5865, 8070, 356, 1250, 5539, 125, 7240, 7949, 3859, 182, 69, 4271, 8481, 8061, 5630, 7854, 3509, 7958, 10, 7502, 4525, 8083, 6462, 7873, 557, 154, 137, 5956, 7809, 8180, 6105, 3357, 5307, 5485, 1285, 6343, 7612, 8046, 7459, 6922, 70, 1713, 855, 8438, 204, 7945, 2085, 4337, 7747, 7633, 7941, 7012, 7843, 2567, 2522, 1872, 620, 558, 8226, 4851, 7973, 8233, 527, 7697, 213, 8214, 8421, 541, 8240, 4466, 6756, 7912, 8060, 5421, 5665, 2155, 3881, 7994, 3195, 7131, 8205, 2555, 6930, 7539, 4152, 7731, 3697, 7968, 607, 271, 5809, 239, 11, 4424, 3692, 8232, 7, 6239, 8295, 3073, 365, 5974, 1922, 1136, 2986, 7853, 2227, 7795, 6839, 1213, 8239, 524, 8124, 5579, 5530, 276, 519, 4813, 5552, 5034, 7704, 7788, 8162, 5308, 2234, 1416, 8472, 8225, 3292, 7842, 8154, 6999, 5342, 4514, 7634, 7613, 820, 8095, 3794, 6161, 102, 8327, 253, 7810, 7443, 1043, 107, 3949, 7741, 25, 254, 4981, 7519, 411, 3838, 7829, 8155, 8208, 5768, 8312, 3293, 6015, 7525, 8203, 71, 6997, 8244, 8326, 5755, 7579, 8343, 2017, 2386, 6849, 5052, 6091, 2583, 2088, 422, 8490, 4769, 469, 514, 6897, 575, 763, 3314, 3883, 8160, 8189, 5672, 7211, 5553, 6266, 8211, 7706, 8350, 805, 455, 5058, 4900, 403, 7775, 7876, 7461, 217, 6908, 8037, 7620, 7982, 1785, 7955, 8434, 8289, 7098, 6018, 5919, 5878, 5245, 8537, 8273, 538, 7838, 3614, 8212, 8193, 8210, 13, 8030, 5800, 842, 8250, 8269, 3580, 6211, 7520, 566, 8143, 8373, 6444, 3337, 8159, 4258, 7708, 8204, 7848, 7567, 151, 7937, 694, 473, 8190, 248, 8264]

    # outside loop
    L_m = list(map(lambda x: x-1, L_m))
    L_f = list(map(lambda x: x-1, L_f))

    longtail_item = np.loadtxt('ml-'+dataset+'/longtail_item.dat', dtype=int)
    longtail_item = list(longtail_item)
    popular_item = np.loadtxt('ml-'+dataset+'/popular_item.dat', dtype=int)
    popular_item = list(popular_item)

    with open('ml-'+dataset+'/Dist/combine_personalized_recommendations_top100.json') as json_file:
        item_choice = json.load(json_file)

    long_Lm = [item_id for item_id in L_m if item_id in longtail_item]
    long_Lf = [item_id for item_id in L_f if item_id in longtail_item]

    # Calculate average ratings and initial count
    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = [rating for rating in X[:, item_id] if rating > 0]
        avg_ratings[item_id] = np.average(ratings) if ratings else 0
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor

    # Obfuscation starts
    X_obf = np.copy(X)
    total_added = 0

    for index, user in enumerate(X):
        print(f"Processing User: {index}")
        rate = sum(1 for rating in user if rating > 0)  # Count of non-zero ratings for the user
        k = rate * p

        # Calculate two proportions of `k`
        if k == 1 :
            k1 = k
            k2 = 0
        else:
            k1 = 0.7 * k
            k2 = 0.3 * k
            print(f"k -> {k}, k1 {k1} & k2 {k2}")

        greedy_index = 0
        added1 = 0
        added2 = 0
        mylist = list(item_choice.values())
        safety_counter = 0
        print(f"User: {index}, No of Ratings: {rate}, p:{p} & k = {k}, k1 = {k1}, k2 = {k2}")

        # --- First Proportion: Add items without checking for long-tail constraint ---
        while added1 < k1 and safety_counter < 100:
            if greedy_index >= len(mylist[index]):
                safety_counter = 100
                continue

            vec = mylist[index]
            if sample_mode == 'greedy':
                movie_id = int(vec[greedy_index])  # Get movie ID for greedy mode
            elif sample_mode == 'random':
                movie_id = int(vec[np.random.randint(0, len(vec))])  # Get movie ID for random mode

            greedy_index += 1

            rating_count = sum(1 if x > 0 else 0 for x in X_obf[:, movie_id])
            if rating_count > max_count[movie_id]:
                continue

            if X_obf[index, movie_id] == 0:
                X_obf[index, movie_id] = avg_ratings[movie_id]  # Assign average rating
                print(f"Added rating (no constraints): {X_obf[index, movie_id]} for movie: {movie_id}")
                added1 += 1

            safety_counter += 1

        # --- Second Proportion: Add only if the movie is in the long-tail items ---
        while added2 < k2 and safety_counter < 200:
            if greedy_index >= len(mylist[index]):
                safety_counter = 200
                continue

            vec = mylist[index]
            if sample_mode == 'greedy':
                #movie_id = int(vec[greedy_index]) #.58 # Get movie ID for greedy mode
                #movie_id = int(not_in_Lm_or_Lf[greedy_index]) #.61
                if T[index] == 0:
                    movie_id = int(long_Lf[greedy_index])
                elif T[index] == 1:
                    movie_id = int(long_Lm[greedy_index])
            elif sample_mode == 'random':
                #movie_id = int(vec[np.random.randint(0, len(vec))])  # Get movie ID for random mode
                if T[index] == 0:
                    movie_id = int(long_Lf[np.random.randint(0, len(long_Lf))])
                elif T[index] == 1:
                    movie_id = int(long_Lm[np.random.randint(0, len(long_Lm))])

            greedy_index += 1

            rating_count = sum(1 if x > 0 else 0 for x in X_obf[:, movie_id])
            if rating_count > max_count[movie_id]:
                continue

            if X_obf[index, movie_id] == 0:
                X_obf[index, movie_id] = avg_ratings[movie_id]  # Assign average rating
                print(f"Added rating (long-tail only): {X_obf[index, movie_id]} for movie: {movie_id}")
                added2 += 1

            safety_counter += 1

        total_added += (added1 + added2)

    # Save the obfuscated data to a file
    output_file = 'ml-'+dataset+'/SBlur/'
    print(output_file)
    with open(output_file + "SBlur_" + rating_mode + "_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(int(np.round(rating))) + "::000000000\n")

    return X_obf

def SmartBlur_Removal():

    sample_mode = list(['random', 'sampled', 'greedy'])[2]
    rating_mode = list(['highest', 'avg', 'pred'])[1]
    removal_mode = list(['random', 'strategic'])[1]
    top = 100
    p = 0.1
    notice_factor = 2
    dataset = ['100k', '1m', 'yahoo'][2]

    if dataset == '100k':
        X = DL.load_user_item_matrix_100k()
        T = DL.load_gender_vector_100k()
        X_filled = DL.load_user_item_matrix_100k_Impute()
        L_m = [751, 327, 24, 688, 472, 888, 317, 948, 315, 178, 305, 179, 359, 186, 10, 883, 294, 342, 5, 471, 1048, 1101, 879, 449, 324, 150, 357, 411, 1265, 136, 524, 678, 202, 692, 100, 303, 264, 339, 1010, 276, 218, 995, 240, 433, 271, 203, 544, 520, 410, 222, 1028, 924, 823, 322, 462, 265, 750, 23, 281, 137, 333, 736, 164, 358, 301, 479, 647, 168, 144, 94, 687, 60, 70, 175, 6, 354, 344, 1115, 597, 596, 42, 14, 521, 184, 393, 188, 450, 1047, 682, 7, 760, 737, 270, 1008, 260, 295, 1051, 289, 430, 235, 258, 330, 56, 347, 52, 205, 371, 1024, 8, 369, 116, 79, 831, 694, 287, 435, 495, 242, 33, 201, 230, 506, 979, 620, 355, 181, 717, 936, 500, 893, 489, 1316, 855, 2, 180, 871, 755, 523, 477, 227, 412, 183, 657, 510, 77, 474, 1105, 67, 705, 362, 1133, 511, 546, 768, 96, 457, 108, 1137, 352, 341, 825, 165, 584, 505, 908, 404, 679, 15, 513, 117, 490, 665, 28, 338, 59, 32, 1014, 989, 351, 475, 57, 864, 969, 177, 316, 463, 134, 703, 306, 378, 105, 99, 229, 484, 651, 157, 232, 114, 161, 395, 754, 931, 591, 408, 685, 97, 554, 468, 455, 640, 473, 683, 300, 31, 1060, 650, 72, 191, 259, 1280, 199, 826, 747, 68, 1079, 887, 578, 329, 274, 174, 428, 905, 780, 753, 396, 4, 277, 71, 519, 1062, 189, 325, 502, 233, 1022, 880, 1063, 197, 813, 16, 331, 208, 162, 11, 963, 501, 820, 930, 896, 318, 1142, 1194, 425, 171, 282, 496, 26, 215, 573, 527, 730, 49, 693, 517, 336, 417, 207, 900, 299, 226, 606, 268, 192, 407, 62, 636, 480, 1016, 241, 945, 343, 603, 231, 469, 515, 492, 664, 972, 642, 781, 984, 187, 149, 257, 40, 141, 978, 44, 845, 85, 244, 512, 182, 1021, 756, 1, 778, 644, 1050, 185, 530, 81, 497, 3, 335, 923, 509, 418, 724, 566, 221, 570, 655, 413, 1311, 340, 583, 537, 635, 686, 176, 296, 926, 561, 101, 173, 862, 680, 652]
        L_f = [332, 321, 877, 906, 827, 882, 337, 278, 872, 304, 87, 93, 1038, 292, 629, 727, 220, 898, 107, 143, 516, 819, 382, 499, 876, 45, 1600, 111, 416, 132, 752, 125, 689, 604, 628, 310, 246, 419, 148, 659, 881, 248, 899, 707, 20, 129, 272, 690, 350, 498, 19, 216, 713, 429, 253, 372, 311, 251, 288, 1023, 392, 1328, 131, 320, 83, 895, 912, 445, 1094, 432, 109, 156, 532, 588, 486, 653, 243, 319, 518, 269, 298, 447, 290, 1386, 988, 423, 723, 279, 696, 568, 919, 1296, 266, 645, 937, 595, 482, 748, 348, 749, 815, 451, 154, 684, 993, 904, 204, 198, 403, 525, 553, 671, 507, 1119, 1036, 65, 420, 169, 9, 427, 832, 894, 729, 654, 1128, 987, 1281, 401, 1089, 942, 190, 406, 170, 402, 878, 291, 710, 297, 721, 151, 367, 954, 856, 64, 312, 385, 581, 370, 39, 648, 691, 746, 89, 1033, 745, 194, 662, 145, 539, 98, 147, 739, 159, 90, 236, 55, 120, 448, 200, 409, 623, 118, 731, 365, 238, 124, 146, 716, 1026, 847, 261, 153, 262, 130, 127, 866, 58, 356, 195, 1086, 1190, 1041, 1073, 889, 405, 950, 891, 611, 582, 66, 1035, 346, 1394, 172, 909, 1013, 785, 925, 488, 704, 1315, 708, 1431, 309, 121, 239, 902, 213, 446, 1012, 1054, 873, 193, 478, 829, 916, 353, 349, 133, 249, 550, 638, 452, 1313, 466, 529, 458, 792, 313, 625, 668, 1061, 742, 1065, 763, 73, 846, 1090, 863, 1620, 890, 1176, 1017, 237, 1294, 245, 508, 387, 53, 514, 422, 1068, 1527, 939, 1232, 1011, 631, 381, 956, 875, 459, 607, 1442, 155, 697, 1066, 1285, 293, 88, 1221, 1109, 675, 254, 209, 649, 140, 48, 613, 962, 1157, 991, 1083, 720, 535, 901, 436, 286, 559, 1085, 892, 102, 211, 285, 1383, 758, 1234, 674, 1163, 283, 637, 735, 885, 630, 709, 938, 38, 142, 219, 953, 275, 1059, 676, 210, 47, 63, 255, 494, 744, 250, 1114, 672, 308, 632, 17, 1203, 762, 1522, 538, 491, 307, 669, 775, 1135, 217, 841, 949, 766, 966, 314, 812, 821, 574, 1039, 1388, 51, 579, 252, 50, 1009, 1040, 1147, 224, 212, 1483, 1278, 122, 934, 702, 443, 86, 1444, 69, 400, 975, 35, 549, 928, 531, 622, 614, 1020, 658, 740, 1468, 22, 1007, 470, 1269, 633, 1074, 1299, 1095, 1148, 280, 1136, 92, 167, 434, 284, 961, 1084, 126, 619, 974, 196, 485, 1152, 673, 627, 345, 29, 605, 929, 952, 714, 1053, 526, 123, 476, 660, 955, 380, 1503, 163, 493, 1197, 431, 504, 886, 1037, 13, 794, 273, 844, 1032, 1025, 106, 91, 533, 421, 699, 869, 78, 1243, 481, 661, 82, 732]

    elif dataset == '1m':
        X = DL.load_user_item_matrix_1m()
        T = DL.load_gender_vector_1m()
        X_filled = DL.load_user_item_matrix_1m_Impute()
        L_m = [589.0, 1233.0, 2706.0, 1090.0, 2959.0, 1250.0, 2427.0, 2490.0, 1208.0, 1266.0, 3654.0, 1748.0, 1262.0, 1396.0, 1374.0, 2700.0, 1265.0, 1089.0, 1222.0, 231.0, 2770.0, 1676.0, 2890.0, 1228.0, 1136.0, 3360.0, 3298.0, 1663.0, 3811.0, 2011.0, 1261.0, 233.0, 3361.0, 2366.0, 1127.0, 1276.0, 3555.0, 1214.0, 3929.0, 299.0, 1304.0, 3468.0, 1095.0, 150.0, 1213.0, 750.0, 3082.0, 6.0, 111.0, 3745.0, 349.0, 541.0, 2791.0, 785.0, 1060.0, 1294.0, 1302.0, 1256.0, 1292.0, 2948.0, 3683.0, 3030.0, 3836.0, 913.0, 2150.0, 32.0, 2826.0, 2721.0, 590.0, 3623.0, 2997.0, 3868.0, 3147.0, 1610.0, 3508.0, 2046.0, 21.0, 1249.0, 10.0, 1283.0, 3760.0, 2712.0, 3617.0, 3552.0, 3256.0, 1079.0, 3053.0, 1517.0, 2662.0, 1953.0, 2670.0, 3578.0, 2371.0, 3334.0, 2502.0, 2278.0, 364.0, 3462.0, 2401.0, 3163.0, 2311.0, 852.0, 2916.0, 1378.0, 3384.0, 524.0, 70.0, 370.0, 3035.0, 3513.0, 2917.0, 3697.0, 24.0, 1957.0, 3494.0, 1912.0, 3752.0, 2013.0, 3452.0, 3928.0, 2987.0, 431.0, 2759.0, 1387.0, 1882.0, 3638.0, 1288.0, 2867.0, 2728.0, 2433.0, 161.0, 3386.0, 517.0, 741.0, 1287.0, 1231.0, 3062.0, 2288.0, 3753.0, 529.0, 3793.0, 3052.0, 2447.0, 1320.0, 3819.0, 1303.0, 922.0, 3022.0, 260.0, 858.0, 493.0, 3006.0, 480.0, 2410.0, 333.0, 1178.0, 3814.0, 2702.0, 1203.0, 2922.0, 1625.0, 3366.0, 3213.0, 2188.0, 2628.0, 3358.0, 2648.0, 3788.0, 953.0, 999.0, 3754.0, 3910.0, 3016.0, 3863.0, 303.0, 3263.0, 1080.0, 786.0, 3764.0, 2105.0, 3543.0, 2607.0, 3681.0, 592.0, 145.0, 2303.0, 1682.0, 1019.0, 3646.0, 1544.0, 235.0, 908.0, 3615.0, 2792.0, 2167.0, 2455.0, 1587.0, 1227.0, 2901.0, 2687.0, 1883.0, 1210.0, 1201.0, 3169.0, 3098.0, 3688.0, 2409.0, 3198.0, 610.0, 1923.0, 1982.0, 165.0, 2403.0, 784.0, 2871.0, 2889.0, 628.0, 2300.0, 417.0, 3671.0, 3100.0, 3914.0, 3608.0, 3152.0, 3429.0, 1794.0, 952.0, 1391.0, 2518.0, 410.0, 3535.0, 2333.0, 1713.0, 2605.0, 707.0, 2795.0, 1965.0, 373.0, 3916.0, 556.0, 3703.0, 95.0, 466.0, 3066.0, 3177.0, 2088.0, 1476.0, 163.0, 3422.0, 58.0, 1244.0, 1689.0, 2002.0, 1711.0, 2259.0, 3524.0, 1371.0, 3104.0, 1693.0, 965.0, 1732.0, 2600.0, 3424.0, 3755.0, 2450.0, 3826.0, 3801.0, 3927.0, 1298.0, 2118.0, 112.0, 2478.0, 471.0, 1673.0, 1246.0, 2734.0, 2529.0, 2806.0, 1948.0, 2093.0, 45.0, 648.0, 3504.0, 2968.0, 1722.0, 1963.0, 2840.0, 1747.0, 1348.0, 3871.0, 3175.0, 2360.0, 1092.0, 3190.0, 1405.0, 367.0, 3248.0, 1702.0, 1734.0, 2644.0, 1597.0, 1401.0, 1416.0, 107.0, 1379.0, 2764.0, 2116.0, 1036.0, 60.0, 2115.0, 1876.0, 1254.0, 2243.0, 2606.0, 3925.0, 3087.0, 1627.0, 3770.0, 3678.0, 3113.0, 3036.0, 3525.0, 1584.0, 2236.0, 3267.0, 954.0, 1205.0, 2470.0, 2686.0, 3397.0, 2015.0, 1377.0, 3740.0, 1594.0, 2456.0, 2038.0, 891.0, 1342.0, 1966.0, 2808.0, 3324.0, 3794.0, 2467.0, 3420.0, 3773.0, 1927.0, 2231.0, 3742.0, 1960.0, 1542.0, 2672.0, 1376.0, 3174.0, 1248.0, 225.0, 1267.0, 3203.0, 1025.0, 2769.0, 1973.0, 2541.0, 3593.0, 2058.0, 3273.0, 154.0, 1179.0, 2009.0, 2423.0, 2676.0, 2793.0, 3505.0, 1920.0, 3357.0, 2580.0, 2542.0, 1701.0, 3252.0, 440.0, 540.0, 1885.0, 2384.0, 1414.0, 1251.0, 1187.0, 2841.0, 2287.0, 2004.0, 1257.0, 1358.0, 2253.0, 3918.0, 2976.0, 1100.0, 2140.0, 2092.0, 2772.0, 3500.0, 1196.0, 3728.0, 555.0, 3564.0, 3099.0, 2863.0, 2492.0, 13.0, 2378.0, 3271.0, 3946.0, 1017.0, 3189.0, 3908.0, 1238.0, 3551.0, 800.0, 1193.0, 3254.0, 3614.0, 448.0, 1779.0, 3477.0, 1388.0, 748.0, 1411.0, 3948.0, 1057.0, 2877.0, 2633.0, 3078.0, 2289.0, 514.0, 3831.0, 535.0, 361.0, 290.0, 1408.0, 1356.0, 2522.0, 2321.0, 1395.0, 1103.0, 2861.0, 1974.0, 2497.0, 1633.0, 2530.0, 1931.0, 125.0, 1735.0, 3159.0, 892.0, 2828.0, 523.0, 3148.0, 296.0, 2882.0, 1639.0, 1665.0, 3834.0, 534.0, 2942.0, 1247.0, 861.0, 2107.0, 3469.0, 1970.0, 3307.0, 432.0, 3879.0, 3930.0, 742.0, 3937.0, 1237.0, 1091.0, 3214.0, 1273.0, 3809.0, 3115.0, 2111.0, 468.0, 3769.0, 2961.0, 3771.0, 246.0, 3094.0, 2907.0, 1016.0, 151.0, 377.0, 450.0, 3538.0, 3717.0, 2694.0, 2745.0, 2389.0, 3865.0, 281.0, 2272.0, 2991.0, 1810.0, 2024.0, 2725.0, 2731.0, 409.0, 2971.0, 1083.0, 2701.0, 1753.0, 1459.0, 2567.0, 673.0, 3516.0, 611.0, 947.0, 1176.0, 1640.0, 172.0, 2671.0, 2041.0, 2723.0, 2471.0, 378.0, 3901.0, 1834.0, 1733.0, 1135.0, 998.0, 2475.0, 292.0, 3347.0, 2121.0, 3952.0, 1219.0, 413.0, 2294.0, 1997.0, 849.0, 2017.0, 2025.0, 3476.0, 1399.0, 2822.0, 2068.0, 180.0, 2076.0, 3700.0, 1783.0, 3326.0, 1760.0, 2437.0, 3893.0, 2594.0, 16.0, 1942.0, 2171.0, 2815.0, 1281.0, 1589.0, 936.0, 3168.0, 2520.0, 3095.0, 3448.0, 1971.0, 1230.0, 3129.0, 3799.0, 3125.0, 3784.0, 3789.0, 3262.0, 1946.0, 2390.0, 1918.0, 3201.0, 3909.0, 2943.0, 2082.0, 3157.0, 2112.0, 3409.0, 1772.0, 1680.0, 3633.0, 2153.0, 720.0, 674.0, 3713.0, 126.0, 585.0, 2353.0, 158.0, 3676.0, 3398.0, 485.0, 765.0, 1284.0, 2089.0, 1148.0, 1147.0, 2183.0, 1037.0, 2393.0, 2250.0, 2524.0, 1617.0, 1457.0, 3135.0, 3142.0, 2935.0, 1461.0, 533.0, 1425.0, 1282.0, 728.0, 3521.0, 1972.0, 1361.0, 551.0, 2016.0, 454.0, 3889.0, 3837.0, 190.0, 2735.0, 2124.0, 2310.0, 23.0, 3548.0, 1466.0, 3743.0, 1124.0, 2033.0, 1590.0, 2138.0, 2716.0, 1649.0, 1189.0, 2135.0, 3243.0, 3359.0, 1339.0, 123.0, 1224.0, 2996.0, 344.0, 1101.0, 515.0, 2428.0, 1873.0, 1392.0, 2583.0, 258.0, 2519.0, 2771.0, 213.0, 451.0, 2906.0, 2313.0, 3253.0, 1343.0, 2941.0, 745.0, 2729.0, 353.0, 1707.0, 2859.0, 2108.0, 1359.0]
        L_f = [920.0, 3844.0, 2369.0, 1088.0, 3534.0, 1207.0, 17.0, 1041.0, 3512.0, 3418.0, 1188.0, 902.0, 2336.0, 3911.0, 1441.0, 141.0, 2690.0, 928.0, 39.0, 2762.0, 906.0, 838.0, 2657.0, 2125.0, 3565.0, 1967.0, 2291.0, 914.0, 932.0, 1620.0, 2160.0, 247.0, 222.0, 261.0, 2881.0, 2145.0, 3072.0, 1028.0, 1956.0, 2080.0, 1286.0, 3798.0, 1959.0, 28.0, 2248.0, 3247.0, 3594.0, 3155.0, 1345.0, 531.0, 1277.0, 593.0, 3044.0, 3083.0, 3005.0, 1380.0, 2020.0, 105.0, 1678.0, 1608.0, 2572.0, 3791.0, 1104.0, 2144.0, 318.0, 1186.0, 1073.0, 595.0, 2724.0, 1641.0, 351.0, 2908.0, 357.0, 3079.0, 1688.0, 3556.0, 3186.0, 2406.0, 224.0, 1962.0, 1480.0, 3251.0, 11.0, 345.0, 3526.0, 1784.0, 951.0, 3668.0, 2485.0, 1958.0, 2739.0, 916.0, 950.0, 2443.0, 3684.0, 904.0, 898.0, 587.0, 552.0, 2143.0, 3481.0, 3097.0, 3067.0, 1449.0, 47.0, 616.0, 3281.0, 1259.0, 661.0, 2348.0, 562.0, 3606.0, 2496.0, 2085.0, 1271.0, 372.0, 2857.0, 3325.0, 1394.0, 1081.0, 1032.0, 918.0, 1409.0, 314.0, 899.0, 733.0, 2245.0, 381.0, 2316.0, 232.0, 2405.0, 2677.0, 1066.0, 2396.0, 2282.0, 1059.0, 2622.0, 1941.0, 959.0, 3479.0, 3124.0, 1197.0, 1777.0, 915.0, 955.0, 1648.0, 3705.0, 3061.0, 34.0, 1285.0, 1.0, 2875.0, 1150.0, 3545.0, 2664.0, 2155.0, 1097.0, 262.0, 3915.0, 971.0, 2186.0, 3702.0, 3105.0, 2280.0, 3604.0, 3515.0, 1513.0, 2331.0, 1500.0, 2803.0, 945.0, 2639.0, 3051.0, 837.0, 3408.0, 457.0, 1801.0, 2506.0, 4.0, 2469.0, 270.0, 46.0, 1235.0, 2355.0, 2346.0, 1357.0, 461.0, 3255.0, 3176.0, 3350.0, 2975.0, 2014.0, 3936.0, 2072.0, 1353.0, 2006.0, 1397.0, 2612.0, 1099.0, 1367.0, 3270.0, 938.0, 2357.0, 94.0, 412.0, 1518.0, 3591.0, 538.0, 2000.0, 2846.0, 708.0, 329.0, 2995.0, 653.0, 1280.0, 5.0, 337.0, 1022.0, 2468.0, 1569.0, 905.0, 1031.0, 900.0, 1541.0, 2926.0, 3730.0, 1900.0, 2718.0, 1021.0, 3185.0, 2746.0, 327.0, 2805.0, 3101.0, 2920.0, 3269.0, 1674.0, 477.0, 3686.0, 2077.0, 2801.0, 581.0, 2133.0, 3724.0, 3296.0, 3554.0, 3478.0, 1479.0, 3720.0, 491.0, 1014.0, 1236.0, 3134.0, 695.0, 2763.0, 1013.0, 1096.0, 1856.0, 2827.0, 248.0, 1875.0, 3211.0, 3672.0, 215.0, 3224.0, 3396.0, 469.0, 1897.0, 3528.0, 2870.0, 917.0, 930.0, 1654.0, 3328.0, 3786.0, 907.0, 3870.0, 1422.0, 2206.0, 2114.0, 2324.0, 2575.0, 919.0, 3467.0, 1047.0, 1806.0, 350.0, 230.0, 2505.0, 48.0, 182.0, 144.0, 170.0, 2141.0, 1916.0, 3081.0, 1191.0, 1086.0, 2598.0, 546.0, 1407.0, 153.0, 2635.0, 2057.0, 2037.0, 1327.0, 3145.0, 446.0, 2193.0, 1337.0, 1913.0, 195.0, 2132.0, 1804.0, 3562.0, 3706.0, 1172.0, 1042.0, 2946.0, 2514.0, 1093.0, 1616.0, 3011.0, 2151.0, 1111.0, 613.0, 1043.0, 2774.0, 2154.0, 2621.0, 52.0, 3060.0, 3723.0, 206.0, 3133.0, 1821.0, 1964.0, 211.0, 2454.0, 532.0, 218.0, 3156.0, 1586.0, 1126.0, 2096.0, 927.0, 2007.0, 778.0, 2097.0, 3117.0, 691.0, 3567.0, 1223.0, 1268.0, 1300.0, 2747.0, 1573.0, 3302.0, 671.0, 3471.0, 3825.0, 1064.0, 1299.0, 252.0, 3004.0, 2091.0, 2337.0, 61.0, 1020.0, 3763.0, 1727.0, 74.0, 3599.0, 3708.0, 465.0, 29.0, 3741.0, 3457.0, 2399.0, 781.0, 69.0, 3635.0, 3808.0, 3249.0, 2732.0, 1621.0, 1686.0, 3435.0, 3857.0, 3299.0, 3426.0, 176.0, 343.0, 2972.0, 2853.0, 272.0, 2788.0, 1393.0, 203.0, 1465.0, 801.0, 1917.0, 2431.0, 3714.0, 2967.0, 3553.0, 79.0, 3951.0, 1683.0, 3071.0, 3102.0, 302.0, 3655.0, 2261.0, 3877.0, 2266.0, 3716.0, 3699.0, 1769.0, 266.0, 1173.0, 2693.0, 3093.0, 1658.0, 277.0, 279.0, 848.0, 839.0, 2365.0, 2738.0, 1264.0, 271.0, 1269.0, 2043.0, 3855.0, 1030.0, 1346.0, 2052.0, 2142.0, 2719.0, 2574.0, 2053.0, 1410.0, 3912.0, 1381.0, 3660.0, 2446.0, 2613.0, 2314.0, 978.0, 348.0, 2168.0, 3466.0, 669.0, 3649.0, 2448.0, 2899.0, 1611.0, 2940.0, 8.0, 1463.0, 26.0, 3557.0, 1994.0, 1758.0, 414.0, 1027.0, 3088.0, 3391.0, 1936.0, 2205.0, 3861.0, 332.0, 3450.0, 2585.0, 3618.0, 425.0, 1605.0, 3827.0, 846.0, 2267.0, 2359.0, 2952.0, 2786.0, 3923.0, 1290.0, 3240.0, 3388.0, 1547.0, 338.0, 3712.0, 3063.0, 242.0, 715.0, 3679.0, 3571.0, 668.0, 1069.0, 2276.0, 1438.0, 2688.0, 2900.0, 168.0, 3539.0, 199.0, 3675.0, 2436.0, 647.0, 724.0, 82.0, 542.0, 1362.0, 117.0, 2109.0, 3246.0, 3019.0, 1904.0, 360.0, 2966.0, 482.0, 2741.0, 334.0, 2100.0, 2173.0, 1615.0, 358.0, 280.0, 3932.0, 369.0, 3547.0, 3739.0, 1788.0, 875.0, 2106.0, 3719.0, 3839.0, 1417.0, 3566.0, 3795.0, 670.0, 520.0, 208.0, 3449.0, 3274.0, 27.0, 3872.0, 2969.0, 2927.0, 2442.0, 113.0, 2084.0, 1848.0, 3882.0, 3790.0, 3926.0, 2820.0, 3922.0, 3046.0, 832.0, 3896.0, 2101.0, 1600.0, 2548.0, 2453.0, 386.0, 239.0, 1015.0, 85.0, 3077.0, 3264.0, 3340.0, 3114.0, 1729.0, 2498.0, 309.0, 1034.0, 2421.0, 3438.0, 2599.0, 405.0, 3461.0, 3813.0, 3238.0, 3399.0, 3921.0, 912.0, 1840.0, 2876.0, 319.0, 40.0, 257.0, 3287.0, 880.0, 754.0, 1874.0, 2241.0, 2553.0, 1699.0, 550.0, 1549.0, 2338.0, 1922.0, 3612.0, 1894.0, 1049.0, 1185.0, 2779.0, 3902.0, 3580.0, 2.0, 2435.0, 73.0, 1012.0, 1275.0, 783.0, 512.0, 1919.0, 3838.0, 2903.0, 507.0, 1896.0, 2263.0, 2320.0, 1515.0, 363.0, 3492.0, 1562.0, 1588.0, 408.0, 3405.0, 307.0, 1199.0, 3268.0, 186.0, 1961.0, 1428.0, 2540.0, 3284.0, 2062.0, 3624.0, 1169.0, 2513.0, 575.0, 380.0, 2696.0, 2070.0, 2130.0, 3897.0, 615.0, 50.0, 3852.0, 415.0, 1797.0, 1660.0, 506.0, 3704.0, 2816.0, 2678.0, 2122.0, 1836.0, 2126.0, 481.0, 87.0, 3577.0, 2990.0, 3200.0, 441.0, 1554.0, 346.0, 1653.0, 2202.0, 2616.0, 283.0, 3584.0, 2417.0, 2284.0, 2042.0, 3454.0, 1582.0, 2568.0, 1669.0, 2048.0, 3613.0, 1911.0, 949.0, 420.0, 1719.0, 2361.0, 41.0, 3949.0, 379.0, 2379.0, 3447.0, 2136.0, 2642.0, 3206.0, 1995.0, 3150.0, 2856.0, 2010.0, 2532.0, 382.0, 2398.0, 1798.0, 1242.0, 2414.0, 2550.0, 1084.0, 131.0, 3055.0, 2630.0, 1949.0, 1954.0, 2352.0, 2110.0, 3181.0, 2021.0, 1344.0, 3685.0, 1398.0, 1312.0, 910.0, 3738.0, 173.0, 1456.0, 3445.0, 986.0, 2848.0, 2722.0, 3696.0, 3864.0, 3707.0, 1171.0, 558.0, 356.0, 2717.0, 3204.0, 2561.0, 934.0, 2704.0, 371.0, 1831.0, 879.0, 2439.0, 3108.0, 2517.0, 1372.0, 1672.0, 807.0, 3616.0, 688.0, 2797.0, 519.0, 1211.0, 1730.0, 1446.0, 1546.0, 2445.0, 2147.0, 3475.0, 1556.0, 1580.0, 1220.0, 2373.0, 501.0, 124.0, 1216.0, 1429.0, 2683.0, 2066.0, 1881.0, 2949.0, 3090.0, 802.0, 1870.0, 407.0, 586.0, 1944.0, 2989.0, 1921.0, 1226.0, 2380.0, 3489.0, 3886.0, 2190.0, 2919.0, 2495.0, 2392.0, 753.0, 1484.0, 1667.0, 2363.0, 3308.0, 1077.0, 1805.0, 2714.0, 3173.0, 216.0, 1694.0, 736.0, 1321.0, 1483.0, 608.0, 1485.0, 1347.0, 2789.0, 25.0, 2699.0, 1792.0, 2065.0, 2709.0, 2860.0, 1845.0, 2752.0, 494.0, 2273.0, 62.0, 2710.0, 866.0, 3841.0, 1566.0, 3153.0, 973.0, 3600.0, 1240.0, 1270.0, 923.0, 2159.0, 896.0, 3258.0, 147.0, 3439.0, 2947.0, 2643.0, 1212.0, 1258.0, 2527.0, 1419.0, 1217.0, 316.0, 1293.0, 2420.0, 3130.0, 2474.0, 2879.0, 991.0, 3317.0, 2713.0, 3440.0, 2463.0, 1619.0, 2539.0, 3070.0, 3040.0, 2163.0, 508.0, 428.0, 1816.0, 2533.0, 2736.0, 1969.0, 3054.0, 2176.0, 288.0, 2794.0, 2239.0, 2290.0, 1234.0, 3735.0, 2166.0, 19.0, 2071.0, 2394.0, 2858.0]
        # --- for this data need to find popular lm, lf
    elif dataset == 'yahoo':
        X = DL.load_user_item_matrix_yahoo()
        T = DL.load_gender_vector_yahoo()
        X_filled = DL.load_user_item_matrix_yahoo_Impute()
        # updated list with each users min rating 20
        L_m = [2587, 3581, 4289, 4049, 132, 916, 7038, 1111, 6790, 1691, 372, 5818, 7266, 1946, 3713, 7661, 2450, 6177, 1487, 4249, 6787, 6262, 4743, 6590, 7262, 8346, 7565, 5073, 5061, 5003, 1442, 7660, 1409, 7064, 2956, 7451, 3425, 1367, 5300, 5908, 7063, 2858, 3210, 292, 7288, 6750, 3123, 4507, 1278, 5373, 5040, 1134, 7895, 6763, 6539, 1483, 2802, 2998, 1066, 4016, 6547, 5164, 3471, 1430, 5532, 1556, 1106, 3239, 3887, 4217, 1415, 7558, 3582, 3534, 6574, 4343, 5729, 762, 6635, 4639, 802, 8568, 3948, 3724, 5577, 4789, 3326, 4481, 6185, 1165, 6811, 5592, 1615, 3755, 6376, 2590, 3258, 6582, 5582, 1376, 1799, 3199, 1555, 5227, 4358, 5265, 4522, 144, 6858, 8287, 1863, 6925, 6292, 6412, 6482, 4004, 5216, 7220, 7759, 2686, 2925, 5130, 2368, 177, 2366, 5013, 3249, 3245, 5937, 578, 2260, 984, 1351, 8141, 3940, 5555, 2115, 4459, 8315, 2693, 1867, 4252, 8136, 3153, 3186, 4056, 3487, 1947, 5935, 769, 1744, 6789, 5814, 4962, 6116, 2677, 8529, 4870, 3570, 6718, 4068, 2947, 1805, 5043, 6455, 6992, 6067, 2930, 3394, 6270, 4244, 7601, 8464, 2648, 5796, 6165, 2815, 5972, 6753, 6857, 3317, 3630, 327]
        L_f = [8569, 8176, 8494, 5099, 8218, 8533, 4931, 126, 760, 7813, 8563, 4468, 8219, 562, 8319, 4636, 1100, 8215, 8379, 1642, 8072, 8323, 3618, 7020, 7864, 7628, 4804, 441, 323, 719, 5302, 7885, 8390, 2315, 8306, 8238, 8301, 8253, 1160, 2405, 1970, 8177, 6944, 5675, 8093, 7656, 1576, 1362, 550, 4819, 6957, 939, 4234, 2258, 6970, 5448, 352, 7651, 7490, 8349, 7600, 54, 7781, 6221, 100, 7478, 92, 8430, 7081, 7587, 5039, 4233, 7592, 2972, 7498, 8506, 4903, 7778, 282, 8235, 6801, 6357, 8474, 303, 7972, 7630, 1621, 6948, 5984, 5391, 52, 36, 6991, 4464, 4893, 7883, 8039, 423, 7732, 3964, 291, 8531, 235, 5225, 1971, 1292, 4280, 5291, 7589, 210, 654, 361, 7557, 8459, 7834, 8134, 6932, 8227, 1101, 587, 7983, 4274, 606, 6967, 7005, 634, 7590, 110, 5841, 7860, 5521, 215, 4010, 542, 7996, 7466, 7990, 7644, 8418, 616, 8425, 8470, 8033, 388, 7756, 382, 5967, 7769, 4486, 5464, 7768, 384, 7705, 6761, 8370, 1908, 8092, 8318, 8398, 5825, 6937, 7772, 8362, 7703, 485, 7835, 8123, 5443, 2023, 8165, 5623, 7737, 5890, 8249, 2906, 4629, 8188, 149, 468, 407, 7987, 4892, 8003, 7964, 5376, 5687, 7655, 563, 6910, 4963, 7999, 7796, 8041, 4741, 4203, 4699, 8485, 6895, 5529, 6193, 7896, 597, 5159, 8027, 7479, 818, 7798, 6587, 601, 3807, 3, 8572, 6904, 2052, 621, 8266, 5850, 7483, 8048, 3941, 8486, 5404, 7936, 5134, 8303, 325, 3831, 8057, 6405, 8157, 373, 3013, 5621, 87, 6894, 8071, 5614, 5605, 8091, 8274, 8206, 329, 6488, 6837, 3826, 2323, 5025, 2494, 8058, 62, 3071, 6174, 5884, 5838, 7707, 5865, 8070, 356, 1250, 5539, 125, 7240, 7949, 3859, 182, 69, 4271, 8481, 8061, 5630, 7854, 3509, 7958, 10, 7502, 4525, 8083, 6462, 7873, 557, 154, 137, 5956, 7809, 8180, 6105, 3357, 5307, 5485, 1285, 6343, 7612, 8046, 7459, 6922, 70, 1713, 855, 8438, 204, 7945, 2085, 4337, 7747, 7633, 7941, 7012, 7843, 2567, 2522, 1872, 620, 558, 8226, 4851, 7973, 8233, 527, 7697, 213, 8214, 8421, 541, 8240, 4466, 6756, 7912, 8060, 5421, 5665, 2155, 3881, 7994, 3195, 7131, 8205, 2555, 6930, 7539, 4152, 7731, 3697, 7968, 607, 271, 5809, 239, 11, 4424, 3692, 8232, 7, 6239, 8295, 3073, 365, 5974, 1922, 1136, 2986, 7853, 2227, 7795, 6839, 1213, 8239, 524, 8124, 5579, 5530, 276, 519, 4813, 5552, 5034, 7704, 7788, 8162, 5308, 2234, 1416, 8472, 8225, 3292, 7842, 8154, 6999, 5342, 4514, 7634, 7613, 820, 8095, 3794, 6161, 102, 8327, 253, 7810, 7443, 1043, 107, 3949, 7741, 25, 254, 4981, 7519, 411, 3838, 7829, 8155, 8208, 5768, 8312, 3293, 6015, 7525, 8203, 71, 6997, 8244, 8326, 5755, 7579, 8343, 2017, 2386, 6849, 5052, 6091, 2583, 2088, 422, 8490, 4769, 469, 514, 6897, 575, 763, 3314, 3883, 8160, 8189, 5672, 7211, 5553, 6266, 8211, 7706, 8350, 805, 455, 5058, 4900, 403, 7775, 7876, 7461, 217, 6908, 8037, 7620, 7982, 1785, 7955, 8434, 8289, 7098, 6018, 5919, 5878, 5245, 8537, 8273, 538, 7838, 3614, 8212, 8193, 8210, 13, 8030, 5800, 842, 8250, 8269, 3580, 6211, 7520, 566, 8143, 8373, 6444, 3337, 8159, 4258, 7708, 8204, 7848, 7567, 151, 7937, 694, 473, 8190, 248, 8264]

    # outside loop
    L_m = list(map(lambda x: x-1, L_m))
    L_f = list(map(lambda x: x-1, L_f))

    longtail_item = np.loadtxt('ml-'+dataset+'/longtail_item.dat', dtype=int)
    longtail_item = list(longtail_item)
    popular_item = np.loadtxt('ml-'+dataset+'/popular_item.dat', dtype=int)
    popular_item = list(popular_item)

    with open('ml-'+dataset+'/Dist/combine_personalized_recommendations_top100.json') as json_file:
        item_choice = json.load(json_file)

    long_Lm = [item_id for item_id in L_m if item_id in longtail_item]
    long_Lf = [item_id for item_id in L_f if item_id in longtail_item]
    all_long_tail_items = set(long_Lm).union(set(long_Lf))
    not_in_Lm_or_Lf = set(longtail_item) - all_long_tail_items
    not_in_Lm_or_Lf = list(not_in_Lm_or_Lf)
    # print(not_in_Lm_or_Lf)

    popular_Lm = [item_id for item_id in L_m if item_id in popular_item]
    popular_Lf = [item_id for item_id in L_f if item_id in popular_item]
    print(f'popular_Lm: {popular_Lm}')
    print(f'popular_Lf: {popular_Lf}')


    popular_items_added_in_males = {}
    popular_items_added_in_females = {}

    # Calculate average ratings and initial count
    avg_ratings = np.zeros(shape=X.shape[1])
    initial_count = np.zeros(shape=X.shape[1])
    for item_id in range(X.shape[1]):
        ratings = [rating for rating in X[:, item_id] if rating > 0]
        avg_ratings[item_id] = np.average(ratings) if ratings else 0
        initial_count[item_id] = len(ratings)
    max_count = initial_count * notice_factor

    # Obfuscation starts
    X_obf = np.copy(X)
    total_added = 0

    for index, user in enumerate(X):
        print(f"Processing User: {index}")
        rate = sum(1 for rating in user if rating > 0)  # Count of non-zero ratings for the user
        k = rate * p #math.ceil(rate * p)  # Total items to add

        # Calculate two proportions of `k`
        if k == 1 :
            k1 = k
            k2 = 0
        else:
            k1 = 0.7 * k
            k2 = 0.3 * k
            print(f"k -> {k}, k1 {k1} & k2 {k2}")

        greedy_index = 0
        added1 = 0
        added2 = 0
        mylist = list(item_choice.values())
        safety_counter = 0
        print(f"User: {index}, No of Ratings: {rate}, p:{p} & k = {k}, k1 = {k1}, k2 = {k2}")

        # --- First Proportion: Add items without checking for long-tail constraint ---
        while added1 < k1 and safety_counter < 100:
            if greedy_index >= len(mylist[index]):
                safety_counter = 100
                continue

            vec = mylist[index]
            if sample_mode == 'greedy':
                movie_id = int(vec[greedy_index])  # Get movie ID for greedy mode
            elif sample_mode == 'random':
                movie_id = int(vec[np.random.randint(0, len(vec))])  # Get movie ID for random mode

            greedy_index += 1

            rating_count = sum(1 if x > 0 else 0 for x in X_obf[:, movie_id])
            if rating_count > max_count[movie_id]:
                continue

            if X_obf[index, movie_id] == 0:
                X_obf[index, movie_id] = avg_ratings[movie_id]  # Assign average rating
                print(f"Added rating (no constraints): {X_obf[index, movie_id]} for movie: {movie_id}")
                added1 += 1

                # Track if the added item belongs to `popular_Lf` for male users & popular_Lm` for female users
                if T[index] == 0 and movie_id in popular_Lf:
                    if movie_id not in popular_items_added_in_males:
                        popular_items_added_in_males[movie_id] = []
                    popular_items_added_in_males[movie_id].append(index)
                elif T[index] == 1 and movie_id in popular_Lm:
                    if movie_id not in popular_items_added_in_females:
                        popular_items_added_in_females[movie_id] = []
                    popular_items_added_in_females[movie_id].append(index)

            safety_counter += 1

        # --- Second Proportion: Add only if the movie is in the long-tail items ---
        while added2 < k2 and safety_counter < 200:
            if greedy_index >= len(mylist[index]):
                safety_counter = 200
                continue

            vec = mylist[index]
            if sample_mode == 'greedy':
                #movie_id = int(vec[greedy_index]) #.58 # Get movie ID for greedy mode
                #movie_id = int(not_in_Lm_or_Lf[greedy_index]) #.61
                if T[index] == 0:
                    movie_id = int(long_Lf[greedy_index])
                elif T[index] == 1:
                    movie_id = int(long_Lm[greedy_index])
            elif sample_mode == 'random':
                #movie_id = int(vec[np.random.randint(0, len(vec))])  # Get movie ID for random mode
                if T[index] == 0:
                    movie_id = int(long_Lf[np.random.randint(0, len(long_Lf))])
                elif T[index] == 1:
                    movie_id = int(long_Lm[np.random.randint(0, len(long_Lm))])

            greedy_index += 1

            rating_count = sum(1 if x > 0 else 0 for x in X_obf[:, movie_id])
            if rating_count > max_count[movie_id]:
                continue

            if X_obf[index, movie_id] == 0:
                X_obf[index, movie_id] = avg_ratings[movie_id]  # Assign average rating
                print(f"Added rating (long-tail only): {X_obf[index, movie_id]} for movie: {movie_id}")
                added2 += 1

            safety_counter += 1

        total_added += (added1 + added2)


    # --- Remove Popular_Lf Items Added by Male Users from Female Users ---
    for movie_id, male_users in popular_items_added_in_males.items():
        print(f"Removing {len(male_users)} ratings for movie {movie_id} from female users...")

        # Track how many ratings we need to remove
        ratings_to_remove = len(male_users)  # Number of ratings to remove

        # Iterate over users to remove exactly `ratings_to_remove` ratings
        removed_count = 0  # Track removed ratings count
        for user_index, user in enumerate(X):
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                if removed_count >= ratings_to_remove:
                    break  # Stop if we have removed enough ratings

                if T[user_index] == 1 and X_obf[user_index, movie_id] != 0:  # Female user
                    X_obf[user_index, movie_id] = 0  # Remove rating
                    removed_count += 1  # Increment the removed count

    for movie_id, female_users in popular_items_added_in_females.items():
        print(f"Removing {len(female_users)} ratings for movie {movie_id} from male users...")

        # Track how many ratings we need to remove
        ratings_to_remove = len(female_users)  # Number of ratings to remove

        # Iterate over users to remove exactly `ratings_to_remove` ratings
        removed_count = 0  # Track removed ratings count

        for user_index, user in enumerate(X):
            rating_count = sum([1 if x > 0 else 0 for x in user])
            if rating_count > 20:
                if removed_count >= ratings_to_remove:
                    break  # Stop if we have removed enough ratings

                # Check if the user is male and has rated the movie
                if T[user_index] == 0 and X_obf[user_index, movie_id] != 0:
                    X_obf[user_index, movie_id] = 0  # Remove rating
                    removed_count += 1  # Increment the removed count

        # print(f"Removed {removed_count} ratings for movie {movie_id}.")


    # --- Ensure Consistent Popular Item Counts ---
    # for item_id in popular_Lm + popular_Lf:
    #     original_count = sum(1 if x > 0 else 0 for x in X[:, item_id])
    #     obfuscated_count = sum(1 if x > 0 else 0 for x in X_obf[:, item_id])
    #
    #     if obfuscated_count > original_count:
    #
    #         excess = obfuscated_count - original_count
    #         #print(f"Reducing {excess} count for item {item_id} to match original...")
    #         for user_index in np.argwhere(X_obf[:, item_id] > 0).flatten():
    #
    #             rating_in_X = sum(1 if x > 0 else 0 for x in X[user_index, :])
    #
    #             if rating_in_X > 20:
    #
    #
    #                 if T[user_index] == 0 and movie_id in popular_Lm:
    #                     if excess <= 0:
    #                         break
    #                     X_obf[user_index, item_id] = 0
    #                     print(user_index)
    #                     excess -= 1
    #                 elif T[user_index] == 1 and movie_id in popular_Lf:
    #                     if excess <= 0:
    #                         break
    #                     X_obf[user_index, item_id] = 0
    #                     print(user_index)
    #                     excess -= 1
    #
    #             # if T[user_index] == 0 and movie_id in popular_Lm:
    #             #     if excess <= 0:
    #             #         break
    #             #     X_obf[user_index, item_id] = 0
    #             #     excess -= 1
    #             # elif T[user_index] == 1 and movie_id in popular_Lf:
    #             #     if excess <= 0:
    #             #         break
    #             #     X_obf[user_index, item_id] = 0
    #             #     excess -= 1

    # Save the obfuscated data to a file
    output_file = 'ml-'+dataset+'/SBlur/'
    print(output_file)
    with open(output_file + "SBlur_Removal_" + rating_mode + "_" + sample_mode + "_" + str(p) + "_" + str(notice_factor) + ".dat", 'w') as f:
        for index_user, user in enumerate(X_obf):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(str(index_user + 1) + "::" + str(index_movie + 1) + "::" + str(int(np.round(rating))) + "::000000000\n")

    return X_obf


# blurMe()
dataset ='100k' # 1m # yahoo
# Personalized_list_User(dataset)# # This will create the lists of indicative items. It goes before the PerBlur function
PerBlur()
# PerBlur_No_Removal()
#SmartBlur()
# SmartBlur_Removal()


