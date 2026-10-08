"""Static stopword sets for the keywords tool.

Maui-owned lowercase function words (articles, prepositions, conjunctions, pronouns,
auxiliary forms and common intensifiers). Entries shorter than two letters are omitted:
the tokenizer never produces them.
"""

ITALIAN = frozenset(
    """
    ad affinche agli ai al all alla alle allo altra altre altri altro anche ancora anzi
    appunto avendo avere avete aveva avevano avrebbe avuto abbastanza
    be che chi ci cio cioe cioè ciò circa co col coi come comunque con contro cosa cosi così cui
    da dagli dai dal dall dalla dalle dallo degli dei del dell della delle dello dentro di
    dopo dove due dunque durante
    ebbe ecc ed egli entrambi era erano esse essendo essere essi
    fa facendo fanno fare fatto fin finche fino forse fosse fra fu fui furono
    gia già gli ha hai hanno ho
    il in infatti inoltre insieme intanto intorno invece io
    la le lei li lo loro lui lungo
    ma magari mai me meno mentre mi mia mie miei mio moltissimo molta molte molti molto
    ne nei nel nell nella nelle nello nessuno no noi non nostra nostre nostri nostro nulla
    oltre ora ossia ovvero
    parecchi per percio perciò perche perchè perché però piu più piuttosto po poca poche pochi
    poco poi possa potrebbe presso propria proprio puo può pure
    qua qual quale quali qualche qualcosa quando quanta quante quanti quanto quasi quella
    quelle quelli quello quest questa queste questi questo qui quindi
    sara sarebbe sarà se sei sembra sempre senza si sia siamo siete solo sono sopra sotto
    sta stata state stati stato stesso su sua sue sugli sui sul sull sulla sulle sullo suo suoi
    tale tali tanta tante tanti tanto te tra tranne tre troppa troppe troppi troppo tu tua
    tue tuo tuoi tutta tutte tutti tutto
    un una uno va vai verso vi voi vostra vostre vostri vostro
    """.split()
)

ENGLISH = frozenset(
    """
    about above after again against all also am an and any are as at
    be because been before being below between both but by
    can could did do does doing down during each etc few for from further
    had has have having he her here hers herself him himself his how
    if in into is it its itself just me more most my myself
    no nor not now of off on once only or other our ours ourselves out over own
    same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up us very
    was we were what when where which while who whom why will with would
    you your yours yourself yourselves
    """.split()
)

FRENCH = frozenset(
    """
    afin ai aie ainsi ait alors apres après as assez au aucun aucune aujourd auquel
    aura aurait aussi autant autre autres aux avaient avais avait avant avec avoir ayant
    beaucoup bien ca ça car ce ceci cela celle celles celui cependant ces cet cette
    ceux chacun chaque chez ci comme comment
    dans de dedans dehors deja déjà depuis des desquels deux devant devrait doit donc
    dont du duquel durant
    elle elles en encore entre es est et etaient étaient etais étais etait était etant
    étant ete été etre être eu eux fait faire fois font
    ici il ils jamais je jusqu jusque
    la laquelle le lequel les lesquels leur leurs lors lorsque lui
    ma mais me meme même memes mêmes mes moi moins mon
    ne ni non nos notre nous on ont ou où oui
    par parce parmi pas pendant peu peut peuvent plus plutot plutôt pour pourquoi pourtant pu puis
    qu quand que quel quelle quelles quels quelque quelques qui quoi
    sa sans se selon sera serait ses si soit son sont sous souvent sur
    ta tandis tel telle tes toi ton toujours tous tout toute toutes tres très trop tu
    un une va vers veut vos votre vous
    """.split()
)

SPANISH = frozenset(
    """
    al algo algun alguna algunas alguno algunos ante antes aqui aquí asi así aun aún aunque
    bastante bien cada casi como cómo con contra cual cuales cuando cuanto cuánto cuyo
    de del demas demás desde donde dónde dos durante
    el ella ellas ello ellos en entre era eran eres es esa esas ese eso esos esta estaba
    estaban estamos estan están estar estas este esto estos estoy
    fue fueron fui ha habia había han has hasta hay haya he hemos hizo hubo incluso
    la las le les lo los luego
    mas más me mediante menos mi mientras mis mismo misma mucho muchos muy
    nada ni ningun ninguna no nos nosotros nuestra nuestro nunca
    otra otras otro otros para pero poco por porque pues
    que qué quien quién quienes
    se sea segun según ser si sí siempre sido siendo sin sino sobre solo sólo son soy su sus
    tal tambien también tampoco tan tanto te tener tengo ti tiene tienen todo toda todos
    todas tras tu tus tuvo un una uno unos usted ustedes ya yo
    """.split()
)

STOPWORDS: dict[str, frozenset[str]] = {
    "italian": ITALIAN,
    "english": ENGLISH,
    "french": FRENCH,
    "spanish": SPANISH,
}
STOPWORDS["all"] = frozenset().union(*STOPWORDS.values())

LANGUAGES = ("italian", "english", "french", "spanish", "all")
