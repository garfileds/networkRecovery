#! /usr/bin/python
# -*- coding: utf-8 -*-

'''@file overview: config leaf_nodes and measure_nodes for convenient of analysising'''

Epsilon = 1


def config():
    configure = {
        '2K_1000': {
            'leaf_nodes': ['281', '413', '162', '1342', '1019', '114', '1125', '1208', '17', '1308', '463', '103', '923', '2', '1243', '1357', '830', '1207', '1130', '167', '1164', '1191', '258', '499', '339', '513', '488', '661', '249', '76', '861', '771', '871', '1395', '1458', '192', '121', '1082', '1425', '196', '223', '1367', '1108', '7', '1406', '1052', '1288', '1232', '946', '1092', '74', '39', '1121', '359', '523', '1270', '255', '310', '1363', '567', '1143', '1132', '1328', '475', '1119', '104', '785', '658', '1429', '879', '541', '155', '176', '862', '1093', '290', '385', '323', '41', '788', '678', '887', '767', '711', '66', '194', '695', '285', '117', '984', '628', '1301', '1077', '1033', '1238', '843', '732', '1225', '1330', '411', '720', '718', '1004', '1454', '989', '319', '364', '1089', '712', '1313', '45', '1224', '649', '367', '1094', '1480', '264', '851', '9', '622', '942', '986', '287', '139', '71', '741', '111', '725', '1051', '809', '229', '983', '493', '1309', '1174', '431', '955', '856', '272', '910', '640', '1030', '1135', '1417', '366', '575', '1137', '1100', '491', '1321', '1488', '1139', '1228', '1264', '1169', '209', '421', '252', '931', '676', '391', '1209', '308', '1263', '1407', '345', '1142', '533', '1167', '185', '683', '81', '835', '498', '951', '1256', '659', '1212', '937', '277', '291', '754', '804', '958', '921', '351', '1260', '1409', '1071', '1081', '1300', '1237', '68', '322', '1027', '1184', '1041', '911', '768', '1344', '65', '226', '752', '1062', '296', '815', '118', '398', '152', '261', '423', '375', '978', '1180', '688', '681', '392', '896', '728', '130', '907', '863', '144', '183', '667', '669', '460', '178', '758', '967', '1341', '374', '390', '371', '1055', '348', '478', '612', '363', '137', '796', '760', '1479', '1040', '1389', '587', '961', '383', '1205', '1456', '173', '207', '202', '534', '1462', '306', '1276', '982', '338', '138', '1336', '1236', '867', '1021', '297', '956', '538', '1038', '874', '992', '899', '1233', '1257', '1354', '850', '485', '935', '639', '295', '94', '317', '457', '276', '325', '1198', '305', '1144', '451', '1440', '37', '465', '1222', '429', '1219', '585', '218', '637', '4', '1405', '99', '938', '445', '358', '462', '435', '458', '1293', '631', '693', '740', '528', '1442', '1162', '455', '1282', '1079', '559', '1156', '1485', '696', '819', '1332', '384', '761', '158', '1196', '51', '537', '1316', '54', '893', '841', '787', '814', '1240', '1213', '602', '845', '1262', '614', '781', '428', '1157', '801', '153', '1016', '690', '63', '353', '988', '608', '459', '380', '1234', '974', '62', '1123', '1133', '304', '1271', '1410', '1359', '789', '73', '1375', '615', '916', '472', '839', '1470', '1192', '656', '973', '654', '1064', '908', '618', '1107', '542', '630', '1152', '873', '632', '416', '655', '616', '1427', '932', '560', '1446', '509', '1170', '424', '239', '79', '265', '1335', '682', '1432', '1244', '782', '133', '1075', '487', '106', '642', '149', '12', '878', '536', '217', '1053', '994', '1267', '1067', '962', '627', '343', '1450', '1422', '212', '243', '170', '1203', '953', '266', '981', '344', '529', '581', '975', '687', '1178', '607', '205', '588', '510', '126', '1384', '591', '1112', '703', '271', '436', '1258', '1176', '214', '389', '486', '461', '990', '1124', '222', '579', '1024', '866', '1061', '621', '570', '267', '1325', '790', '1056', '1392', '329', '876', '799', '508', '563', '419', '1049', '764', '748', '246', '1364', '1101', '808', '292', '370', '727', '774', '1298', '1106', '1281', '95', '470', '48', '520', '1115', '1459', '1358', '1412', '1211', '477', '968', '439', '131', '539', '1183', '1109', '716', '1388', '224', '420', '864', '949', '972', '1420', '98', '1015', '1414', '1253', '1433', '1014', '770', '154', '880', '811', '1299', '3', '42', '1158', '34', '204', '1401', '738', '10', '146', '1473', '803', '964', '522', '1312', '191', '1295', '210', '206', '1285', '802', '123', '593', '269', '912', '1468', '702', '11', '1460', '16', '119', '692', '1009', '985', '1245', '514', '735', '1073', '701', '798', '582', '313', '187', '67', '900', '1387', '1445', '820', '1047', '1002', '1185', '1287', '1104', '818', '244', '832', '610', '746', '1465', '1441', '834', '773', '633', '553', '763', '417', '625', '611', '278', '1314', '1326', '853', '562', '40', '59', '747', '1383', '361', '245', '1310', '1323', '1113', '831', '1379', '399', '944', '294', '116', '652', '1226', '1451', '448', '193', '778', '1280', '1393', '397', '469', '166', '256', '27', '405', '1150', '237', '471', '674', '484', '15', '248', '606', '1274', '1105', '340', '30', '1042', '495', '710', '235', '382', '827', '1476', '1175', '346', '1149', '635', '476', '660', '1023', '28', '236', '141', '859', '247', '691', '1279', '574', '1045', '251', '605', '706', '617', '412', '875', '283', '354', '505', '1452', '489', '1145', '554', '1083', '1154', '365', '1250', '855', '143', '14', '253', '1340', '58', '257', '698', '733', '274', '1428', '22', '775', '108', '168', '1160', '512', '1348', '1229', '1181', '670', '200', '903', '689', '1136', '1277', '1337', '905', '680', '526', '327', '1475', '753', '77', '997', '869', '309', '128', '23', '604', '954', '759', '1351', '201', '1302', '535', '388', '134', '1373', '596', '105', '1111', '36', '64', '1291', '882', '107', '757', '6', '926', '387', '1241', '1278', '777', '624', '515', '1018', '110', '836', '61', '597', '1381', '298', '922', '965', '434', '1304', '726', '96', '87', '175', '1491', '714', '109', '481', '480', '708', '1259', '800', '238', '915', '868', '120', '492', '1477', '301', '906', '1402', '447', '872', '743', '1284', '634', '722', '865', '482', '883', '352', '1066', '598', '1365', '519', '1345', '600', '586', '172', '20', '85', '885', '1394', '1398', '638', '1478', '1448', '328', '506', '719', '163', '1292', '18', '1463', '1251', '331', '360', '1148', '998', '619', '647', '556', '894', '1482', '1380', '572', '1416', '43', '60', '1017', '794', '1086', '449', '948', '1377', '452', '1138', '1068', '362', '425', '959', '330', '1382', '148', '1390', '672', '97', '318', '490', '1044', '823', '1153', '47', '1408', '466', '1215', '320', '1350', '969', '1457', '898', '44', '697', '707', '1054', '1484', '1397', '418', '1218', '700', '1122', '987', '729', '464', '1327', '228', '1168', '909', '213', '1078', '1362', '532', '1355', '1182', '566', '1058', '623', '797', '1230', '524', '731', '1177', '1131', '1201', '1356', '199', '786', '142', '406', '396', '546', '80', '1118', '1374', '50', '1028', '21', '653', '93', '901', '1074', '72', '1419', '113', '446', '1396', '437', '677', '33', '1031', '897', '1043', '189', '941', '195', '934', '1088', '1187', '840', '1231', '1366', '1371', '1483', '303', '1173', '580', '1070', '1001', '1189', '1246', '1141', '620', '603', '89', '101', '500', '332', '860', '263', '1', '402', '407', '1193', '715', '671', '609', '336', '395', '1195', '233', '511', '963', '1461', '1434', '590', '848', '198', '939', '337', '215', '1116', '1360', '288', '1411', '326', '1266', '952', '5', '25', '1413', '890', '52', '1025', '929', '1159', '793', '312', '584', '945', '646', '369', '1072', '717', '19', '454', '723', '1333', '190', '791', '299', '1117', '950', '648', '1369', '1097', '497', '1035', '895', '1239', '1126', '849'],
            'measure_nodes': ['805', '1134', '49', '300', '56', '240', '443', '765', '936', '1036', '891', '357', '547', '1455', '1349'],
            'hop_average_mean': [8, 6, 6, 6, 7, 7],
            'hoplist_raw': [],
            'hoplist_mean': [],
            'hoplist_contrast_mean': [],
            'hoplist_gmm': [],
            'predict_result': []
        },
        '2K_3000': {
            'leaf_nodes': [],
            'measure_nodes': ['1527', '2086', '51', '488', '680', '2135', '1119', '995', '1431', '1807', '692', '1023', '1566', '1696', '2026', '1189', '751', '1364'],
            'predictResult': []
        }
    }

    return configure


def configNodeLevel():
    nodeLevel = {
        '2K_1000': {
            'file': './topo/2K_2000TopoGen.txt',
            'path': './cache/path_2K_1000.json',
            'path_extract': './cache/path_extract_2K_1000.json',
            'shared_path': './cache/sharedPath_2K_1000.json',
            'shared_path_predict': './cache/shared_path_predict_2K_1000.json',
            'sharedPath_gmm': './cache/sharedPath_gmm_2K_1000.json',
            'sharedPath_hop': './cache/sharedPath_hop_2K_1000.json',
            'numOfLeafnode': 1000,
            'numOfMeasurenode': 15,
            'n_components': 5,
            'likehood': './cache/likehood_2K_1000.json'
        },
        '2K_2000': {
            'file': './topo/2K_2000TopoGen.txt',
            'path': './cache/path_2K_2000.json',
            'sharedPath': './cache/sharedPath_2K_2000.json',
            'sharedPath_gmm': './cache/sharedPath_gmm_2K_2000.json',
            'sharedPath_hop': './cache/sharedPath_hop_2K_2000.json',
            'numOfLeafnode': 1200,
            'numOfMeasurenode': 12,
            'n_components': 12,
            'C_likehood': 10,
            'likehood': './cache/likehood_2K_2000.json'
        },
        '2K_3000': {
            'file': './topo/2K_3000TopoGen.txt',
            'path': './cache/path_2K_3000.json',
            'sharedPath': './cache/sharedPath_2K_3000.json',
            'sharedPath_gmm': './cache/sharedPath_gmm_2K_3000.json',
            'sharedPath_hop': './cache/sharedPath_hop_2K_3000.json',
            'numOfLeafnode': 1800,
            'numOfMeasurenode': 18,
            'n_components': 18,
            'C_likehood': 15,
            'likehood': './cache/likehood_2K_3000.json'
        },
        'param': {
            'Epsilon': Epsilon
        }
    }

    return nodeLevel
