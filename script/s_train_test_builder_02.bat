cd ../train_test_builder_for_DDQuote
python train_test_builder_01.py --contract i1705 --tradeDir long --datelist 20170103,20170104,20170105,20170106,20170109
python train_test_builder_01.py --contract i1705 --tradeDir short --datelist 20170103,20170104,20170105,20170106,20170109
python train_test_builder_01.py --contract j1705 --tradeDir long --datelist 20170103,20170104,20170105,20170106,20170109
python train_test_builder_01.py --contract j1705 --tradeDir short --datelist 20170103,20170104,20170105,20170106,20170109
cd ../script