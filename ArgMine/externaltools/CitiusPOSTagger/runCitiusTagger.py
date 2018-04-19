import subprocess

# subprocess.call(['./test.sh'])

# subprocess.Popen('/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/nec.sh pt /home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTest/test1.txt > /home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTestOutput/outputTest1.txt', shell=False)

# subprocess.call(['/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/nec.sh', 'pt', '/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTest/test1.txt', '>', '/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTestOutput/outputTest1.txt'])
# subprocess.call(["/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/nec.sh", "pt", "/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTest/test1.txt > /home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTestOutput/outputTest1.txt"])

# subprocess.Popen(['/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/nec.sh', 'pt', '/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTest/test1.txt', '>', '/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTestOutput/outputTest1.txt'], shell=True)

### Este programa tem de estar no mesmo diretorio que o nec.sh
### caso contrario, ele nao encontra determinados ficheiros necessarios para a tarefa de PoS Tagging

a= []

f= open('/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTestOutput/outputTest4.txt', 'w') 
subprocess.call(['/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/nec.sh', 'pt', '/home/rocha/Documentos/argmine/NLP/CitiusTagger/CitiusTools/test/myTest/test4.txt'], stdout=a)

print a

print "The End!"
