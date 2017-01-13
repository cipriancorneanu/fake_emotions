function sqlite_test_md5_and_packaging ()
  mksqlite( 'open', '' ); % Dummy Datenbank im Speicher erzeugen
  
  % MD5 Test Suite
  % (ref: http://tools.ietf.org/html/rfc1321)
  
  clc
  fprintf( 'MD5 Test Suite:\n\n' );
  q = mksqlite( 'select MD5(?) as MD5', 'a' );
  fprintf( '"a":\n%s\n%s\n\n', lower( q.MD5 ), '0cc175b9c0f1b6a831c399e269772661' );
  q = mksqlite( 'select MD5(?) as MD5', 'abc' );
  fprintf( '"abc":\n%s\n%s\n\n', lower( q.MD5 ), '900150983cd24fb0d6963f7d28e17f72' );
  q = mksqlite( 'select MD5(?) as MD5', 'message digest' );
  fprintf( '"message digest":\n%s\n%s\n\n', lower( q.MD5 ), 'f96b697d7cb7938d525a2f31aaf161d0' );
  q = mksqlite( 'select MD5(?) as MD5', 'abcdefghijklmnopqrstuvwxyz' );
  fprintf( '"abcdefghijklmnopqrstuvwxyz":\n%s\n%s\n\n', lower( q.MD5 ), 'c3fcd3d76192e4007dfb496cca67e13b' );
  q = mksqlite( 'select MD5(?) as MD5', 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789' );
  fprintf( '"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789":\n%s\n%s\n\n', lower( q.MD5 ), 'd174ab98d277d9f5a5611c2c9f419d9f' );
  q = mksqlite( 'select MD5(?) as MD5', '12345678901234567890123456789012345678901234567890123456789012345678901234567890' );
  fprintf( '"12345678901234567890123456789012345678901234567890123456789012345678901234567890":\n%s\n%s\n\n', lower( q.MD5 ), '57edf4a22be3c955ac49da2e2107b67a' );
  
  
  % Packaging Time f�r Zufallszahlen
  
  mksqlite( 'open', '' ); % Dummy Datenbank im Speicher erzeugen
  fprintf( 'BLOSC Test Suite:\n\n' );
  
  compressor = 'lz4hc';   % blosclz, lz4, lz4hc
  
  % Nur typisierte BLOBs lassen die Komprimierung zu!
  mksqlite( 'typedBLOBs', 1 );
  mksqlite( 'compression_check', 1 );
  
  fprintf( '\n1.000.000 Zufallszahlen:\n' );
  
  data = randn( 1e6, 1 );
  mksqlite( 'compression', 'lz4', 9 );
  q = mksqlite( 'select BDCPackTime(?) as t_pack', data );
  
  for level = 9:-1:0
    mksqlite( 'compression', compressor, level );
    q = mksqlite( 'select BDCPackTime(?) as t_pack, BDCUnpackTime(?) as t_unpack, BDCRatio(?) as ratio', data, data, data );
    fprintf( 'Level %d: pack(%gs), unpack(%gs), ratio(%g%%)\n', level, q.t_pack, q.t_unpack, q.ratio*100 );
  end
  
  data = cumsum( data );
  
  fprintf( '\nDieselben Zufallszahlen, kumuliert:\n' );
  
  for level = 9:-1:0
    mksqlite( 'compression', compressor, level );
    q = mksqlite( 'select BDCPackTime(?) as t_pack, BDCUnpackTime(?) as t_unpack, BDCRatio(?) as ratio', data, data, data );
    fprintf( 'Level %d: pack(%gs), unpack(%gs), ratio(%g%%)\n', level, q.t_pack, q.t_unpack, q.ratio*100 );
  end
  
  fprintf( '\nDieselben Zufallszahlen, erste H�lfte konstant 0:\n' );
  
  data( 1:500000 ) = 0;
  for level = 9:-1:0
    mksqlite( 'compression', compressor, level );
    q = mksqlite( 'select BDCPackTime(?) as t_pack, BDCUnpackTime(?) as t_unpack, BDCRatio(?) as ratio', data, data, data );
    fprintf( 'Level %d: pack(%gs), unpack(%gs), ratio(%g%%)\n', level, q.t_pack, q.t_unpack, q.ratio*100 );
  end
  
  % Packaging Time, Bilddaten
  % Bildschirmausschnitt (figure) als RGB-Matrix erzeugen...
  fprintf( '\nBilddaten:\n' );
  
  h = figure;
  set( h, 'units', 'normalized', 'position', [0.5,0.5,0.2,0.2] );
  x = linspace(0,2*pi,20);
  plot( x, sin(x), 'r-', 'linewidth', 2 );
  legend on
  F = getframe(h);
  delete(h);
  data = F.cdata;

  level = 9;
  mksqlite( 'compression', compressor, level );
  q = mksqlite( 'select BDCPackTime(?) as t_pack, BDCUnpackTime(?) as t_unpack, BDCRatio(?) as ratio', data, data, data );
  fprintf( 'Level %d: pack(%gs), unpack(%gs), ratio(%g%%)\n', level, q.t_pack, q.t_unpack, q.ratio*100 );
  
  mksqlite( 0, 'close' ); % Alle Datenbanken schlie�en
end