from mpmath import mpf

GAUSS_4_COEFF = [
    mpf(1.7724538509055160272981674833411451827975494561224),
    mpf(-1.3293403881791370204736256125058588870981620920918),
    mpf(5.8158641982837244645721120547131326310544591529016),
    mpf(-47.980879635840726832719924451383344206199288011438),
    mpf(584.7669705618088582737740792512345075130538226394),
    mpf(-9443.9865745732130611214513799074372963358192356263),
    mpf(190060.22981328591285506920902063717558875836211698),
    mpf(-4581809.1115702853991847041460332176258004248010343),
    mpf(128720199.72817770543334528210262070767483068425406),
    mpf(-4129773074.6123680493198278007924143712341511198177),
    mpf(148981563666.64117737921278791358634844227200164742),
    mpf(-5969420380552.0089936261851157193802796301258841911),
    mpf(263027585518072.89628165378166138519357120242177217),
    mpf(-12640498773262772.457843323084073107667969901000166),
    mpf(657983105786803244.90380869267987694378807609670509),
    mpf(-36879953079350321876.85847722470710269932166522032),
    mpf(2214525932561613858949.4863746023343073983306162764),
    mpf(-141827359357438652290073.72296166420453999602696888),
    mpf(9650169909612388299570432.8998499019172422296683409),
    mpf(-695193161251681920265106843.77208174995922641360745),
    mpf(52860749998674764012158061633.319666062524678424676),
    mpf(-4230747883822505219687364861438.1918445042072982036),
    mpf(355527052282584387495318899435629.87147850696557063),
    mpf(-31297973874311423416571606157925720.968526390371267),
    mpf(2880391658120223186306355629221601507.8846943638556),
    mpf(-276604010929285032580999331074150392802.16719976106),
    mpf(27668380054782232634039769626580639772316.782491483),
    mpf(-2878280091809985033735526034765124887425731.9564057),
    mpf(310931346703651865385411870487707553687177954.82636),
    mpf(-34832352158740999919684717732653100518662737266.97),
    mpf(4041423659217924515681419374931075987677844091400.3),
    mpf(-4.8506861548500298844263358513789680584362381945411e+50),
    mpf(6.0159877115815800324428188781750873380996313545578e+52),
    mpf(-7.7018315407588728006250906229000606762534598682328e+54),
    mpf(1.0168116567229820671707610451041955106039034701038e+57),
    mpf(-1.3830817413554105360809144701370853648878666986562e+59),
    mpf(1.9366025799270237943749637820398647369606983503476e+61),
    mpf(-2.7891002696719264984528616901472781667781949552472e+63),
    mpf(4.1284188794308338347822983662252375969962242301386e+65),
    mpf(-6.2759906234424502815296285740404813854336831806242e+67),
    mpf(9.7917221208121178986140073808607835515363145773702e+69),
    mpf(-1.5668546561980025000812646079020094069693140454508e+72),
    mpf(2.5699214316390451720082884684964207862523481263331e+74),
    mpf(-4.3179162472637654898760190959929107152271284013314e+76),
    mpf(7.4275519537449147844316180756355324945171200199038e+78),
    mpf(-1.3073729363916674173063719749464476445765884088367e+81),
    mpf(2.3534844441359347751868999416386513767016487970597e+83),
    mpf(-4.3307869332384395504037661426058055519677414965044e+85),
    mpf(8.142556119946584860938830936558696594816848973037e+87),
    mpf(-1.5634954059907638838848617464152374813567964852053e+90),
    mpf(3.0646855200527958269969117522358277490815246304753e+92),
    mpf(-6.1298217291526582416154406973763960610673259557491e+94),
    mpf(1.250572043637466598475727048044073879093230658328e+97),
    mpf(-2.6013668185079546946311833533818685846288857925285e+99),
    mpf(5.5152589561838789463256963735241977700499918920761e+101),
    mpf(-1.1913711426123930871192548579954111392094807485779e+104),
    mpf(2.621176072382436094307957016365528891681287488043e+106),
    mpf(-5.8717792937251282244201273952135696553781473005174e+108),
    mpf(1.3388416071498515428603462029284595741372785088365e+111),
    mpf(-3.1062827203173780181914515398876188127918086759044e+113),
    mpf(7.3312155052890517951840995655772663505402674512187e+115),
    mpf(-1.7595818591649292634482230445036944642897118145317e+118),
    mpf(4.2935925890066812249019845781508496171972544538864e+120),
    mpf(-1.0648620762711451185521791037692458342270285956883e+123),
    mpf(2.683577220727848723570071674987222460748310541048e+125),
    mpf(-6.8702673285887613302690261885451726405688382516838e+127),
    mpf(1.7863475766527209999759725934102510816915407731452e+130),
    mpf(-4.7161575666441520281082067480127729583986928135779e+132),
    mpf(1.2639822443043824952097800136889085878467143655037e+135),
    mpf(-3.4381690938823012981113853089744497185721768474708e+137),
    mpf(9.4897150743752104007836496726240142000147694036816e+139),
    mpf(-2.6572204642941544250053456792819700043231496735527e+142),
    mpf(7.546782912393762541434453119335720017486528734314e+144),
    mpf(-2.1735510142102843697674523043938860196938341781749e+147),
    mpf(6.3469892538265516723260047273745114227106944996807e+149),
    mpf(-1.8787722890251975605252206593501291262365926788505e+152),
    mpf(5.6365022722356938523875559142996653519788145424079e+154),
    mpf(-1.7135515917558090710010025910241856014043386353194e+157),
    mpf(5.2779036671840223049399149998343978240178057216524e+159),
    mpf(-1.6467560508417996175808573399008621944906188453365e+162),
    mpf(5.20390350403985321026965739946234024042108904678e+164),
    mpf(-1.6652973055844571000012002720631290837888264680163e+167),
    mpf(5.3957155838715908019093767961435683957212662465986e+169),
    mpf(-1.7698434679760011046299073003712001648362957017721e+172),
    mpf(5.8760383354185358103270627646699176901283621134282e+174),
    mpf(-1.974400728097705254644690092424074725509807273307e+177),
    mpf(6.7131346616422063865990374209359608174361508170874e+179),
    mpf(-2.3093761954554503953209878300211165208602551582403e+182),
    mpf(8.036825982019807328875729778118089565024443086059e+184),
    mpf(-2.8290304717326184140956799919799673984298822621615e+187),
    mpf(1.0071584231907432605715128744781348935693583343483e+190),
    mpf(-3.6258533310490266234910099343472041444508083573629e+192),
    mpf(1.319840171088783590624288684118151443396489221496e+195),
    mpf(-4.8571182683301985153998520582551294206864510197201e+197),
    mpf(1.8068867494220386099509423049160537971237163201685e+200),
    mpf(-6.794036826780766913305802351403171439327051551964e+202),
    mpf(2.5817870725894006517977150951140754842211493323366e+205),
    mpf(-9.9142619814550966524381587782608679580291274425641e+207),
    mpf(3.8468095232585172085183456939150503968068169767514e+210),
    mpf(-1.5079784756137270678392560449366233544149147307727e+213),
    mpf(5.9717078618160302181735418821523987302346434524649e+215),
    mpf(-2.3887274890917226023452942791620645841636363511106e+218),
    mpf(9.6506346976576984107251613359235028365346323023362e+220),
    mpf(-3.9375292282562170998652643092412740371811532897825e+223),
    mpf(1.6222904376850344467101129237557888548140955963997e+226),
    mpf(-6.7488440986581493722031204851757784780876958307082e+228),
    mpf(2.8345622726918377704010573239659390535690828693868e+231),
    mpf(-1.2018742720484842508961025183801840881926385834626e+234),
    mpf(5.1441053478586270719082976746754087435817287371214e+236),
    mpf(-2.2222889054952103259846041479667124171957259547718e+239),
    mpf(9.6893311476572189674514639445085911420927194168474e+241),
    mpf(-4.2633711734228767322932890779077294848933782509765e+244),
    mpf(1.8929653503602935830449098145803928805526034258772e+247),
    mpf(-8.4806104089072807582524600120069685380314522375377e+249),
    mpf(3.8332916983156231874719425337167024566148087235528e+252),
    mpf(-1.748006014160391449290602351521775342188895917997e+255),
    mpf(8.0409406827680265541149577093935839302332868841601e+257),
    mpf(-3.731048021295920526407764190990093415320105584038e+260),
    mpf(1.746154188254761755006247250054069163878805091967e+263),
    mpf(-8.2419578202970293551686680862583611217325742444891e+265),
    mpf(3.9232234346977628294937558132345190087017161689658e+268),
    mpf(-1.8831715661555544500753674210704197513607619948815e+271),
    mpf(9.1146661489367045765481845905681924830462126922344e+273),
    mpf(-4.4480126579137273010663466910789277377782776853512e+276),
    mpf(2.1884491309959202457978345055593671985659717640014e+279),
    mpf(-1.0854838996687624174372007017644794866919133907753e+282),
    mpf(5.427484110480697132567993818387978652476631948721e+284),
    mpf(-2.73548404375378994160835885293497266591900484447e+287),
    mpf(1.3896419224537441601999854087686200301423188301598e+290),
    mpf(-7.1150474360981964806983555246051466834001923126757e+292),
    mpf(3.6714055253772622582508170181319745644807496190655e+295),
    mpf(-1.9091518926934590685399949876939977581171910275798e+298),
    mpf(1.0004064392253083102051195326140849047861246734009e+301),
    mpf(-5.2822024130065766734644336765369142043142412525118e+303),
    mpf(2.8101612482852432300778928825240368842000512394776e+306),
    mpf(-1.5062620410878252893396954622155644590250285757558e+309),
    mpf(8.1338980877956400822113411409343908488339976010982e+311),
    mpf(-4.4248850883999949691759743872803940290148671438894e+314),
    mpf(2.4248610767317211561245707901774128865633483626076e+317),
    mpf(-1.338536398138698198978368064869245820287032486888e+320),
    mpf(7.442334080958205130866100281819339041821987432409e+322),
    mpf(-4.1677466722200042254399675732458608173513864833588e+325),
    mpf(2.3506311358785905873958112886041861657440201001345e+328),
    mpf(-1.3351708136640178378571882724758941976644517246741e+331),
    mpf(7.6372465943047270334721719804513062634634588886633e+333),
    mpf(-4.3990935413191484853853166064570926731321771308643e+336),
    mpf(2.5514968520483663226287167754948044546598466340715e+339),
    mpf(-1.4900871794373278118351840087092489382787721682092e+342),
    mpf(8.7617881262661211819597249704000142933429444760378e+344),
    mpf(-5.1870226737099108510683885233743104751537882788638e+347),
    mpf(3.0914914486444754167910149018737059147440335831442e+350),
    mpf(-1.8549102242766619610452221415293653056638774878017e+353),
    mpf(1.1203749279806578211586219624087770855682138967645e+356),
    mpf(-6.8119344824620064476356547972278356987393271694802e+358),
    mpf(4.1689370782723443518285774030534219468282997504192e+361),
    mpf(-2.5680854124919493094893963992038259196750076219022e+364),
    mpf(1.5922253023094917062551060049890605501308108554237e+367),
    mpf(-9.9355619481294914393108227023103201127788456046043e+369),
    mpf(6.2395800659029225043845351577915565544965364582624e+372),
    mpf(-3.9434440336320899610847261467768948346897096101901e+375),
    mpf(2.5080488902839168656715784140039181315571929200947e+378),
    mpf(-1.6051629732392702284251471643650573118897628967779e+381),
    mpf(1.0337323860761513200235189643472282285667845729458e+384),
    mpf(-6.698633426147053014702710719597232811982176614801e+386),
    mpf(4.3675396278422054575969053588910524880746290713409e+389),
    mpf(-2.8651258483173406083901953559114344824155388282045e+392),
    mpf(1.8909960047351450304860608183142993702970376245314e+395),
    mpf(-1.2556298396411635538880908776458069718444648556123e+398),
    mpf(8.3876633837065279490596742095881711883036968633854e+400),
    mpf(-5.6365470172089984971288217478708083824950619713231e+403),
    mpf(3.810330650752476553170100246362495308368349370829e+406),
    mpf(-2.5910415544882224442107259311240685750207774333452e+409),
    mpf(1.7722837214162573272287014092403067602643935410539e+412),
    mpf(-1.2193388836453160364685802199969664859177998562171e+415),
    mpf(8.4378776325360889256400091129057320897272921515676e+417),
    mpf(-5.8727989945778287611979847997643019875877227401717e+420),
    mpf(4.1109843223365592997225149194293285687346484794387e+423),
    mpf(-2.8941503823500323934423611919007872671712634390523e+426),
    mpf(2.0490706651576811509858157787584198884743914724197e+429),
    mpf(-1.4589468990838827686687345739275819372134111741372e+432),
    mpf(1.0446120586894729118783500746595400319695407898954e+435),
    mpf(-7.5212501075942611591530657730490194265908581709815e+437),
    mpf(5.4454160720607763742590688768266647174043634209753e+440),
    mpf(-3.9642852177392290561014802532678316956831528178611e+443),
    mpf(2.9018729381563836278803709155561286212062187450907e+446),
    mpf(-2.135790246832847632756282238596265622485971075652e+449),
    mpf(1.580493394713754154883329325441230900420354426511e+452),
    mpf(-1.175893424544284616985155106347495473993895111858e+455),
    mpf(8.7957297262331855421480420500712746891767554059174e+457),
    mpf(-6.6144236578167453419236694789633210713192104491999e+460),
    mpf(5.000530394876529807752222772160003479868920412268e+463),
    mpf(-3.8004227356967184517836045642608650530910072708645e+466),
    mpf(2.9035378154736042124691261668006300445656431916877e+469),
    mpf(-2.2299283254617674403740057450387652238250327496223e+472),
    mpf(1.7215132881134127955438885202436833459574503794845e+475),
    mpf(-1.3359009327809626116989634297418684444758505615968e+478),
    mpf(1.0420078394349650499923280783750467050653132180859e+481),
    mpf(-8.1693811315193430591371868677628423750739235985711e+483),
    mpf(6.4375032762627405099298727108382855599005816211422e+486),
    mpf(-5.0985268567471115745950242493182594936632200682472e+489)
]