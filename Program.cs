using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ConsoleApp3
{
    class Program
    {
        static void Main(string[] args)
        {
            Test();
            Console.ReadKey();
        }

        private static void Test()
        {
            using (Py.GIL()) //Initialize the Python engine and acquire the interpreter lock
            {
                try
                {
                    //  Importaçoes //
                    dynamic site = Py.Import("site");
                    dynamic sys = Py.Import("sys");
                    site.addsitedir(@"D:\Usuario\Documents\NinjaTrader 8\bin\TF\netppo");
                    dynamic gym = Py.Import("gym");
                    dynamic np = Py.Import("numpy");    // Numpy para trabalhar com arrays
                    dynamic PPO = Py.Import("PPO");
                    dynamic plt = Py.Import("matplotlib.pyplot");
                    //  Configurações   //
                    int EP_MAX = 700;   // Qantidade total de episódios
                    int EP_LEN = 200;   // Quantas sequencias vão acontecer dentro de cada episódio
                    double GAMMA = 0.9; // Avanço (?)
                    int BATCH = 64;     // Tamanho do pacote à entrar para treinamento em cada etapa (?)

                    // Implementaçao do ambiente   //
                    dynamic env = gym.make("Pendulum-v0").unwrapped;    // Instancia o ambiente pendulo
                    dynamic ppo = PPO.PPO();                            // Instancia a classe PPO
                    List<double> all_ep_r = new List<double>();         // Cria um array para a recompensa de todos os episodios

                    //  Loop de episódios //
                    for (int epmax = 0; epmax < EP_MAX; epmax++)    // EP_MAX: quantidade de episodios 
                    {
                        dynamic s = env.reset();     // Redefine o ambiente e armazena o estado atual em s
                        // Cria quatro arrais para o episódio:
                        List<PyObject> buffer_s = new List<PyObject>(); // buffer_s: buffer do estado
                        List<PyObject> buffer_a = new List<PyObject>(); // buffer_a: buffer da ação
                        List<double> buffer_r = new List<double>();     // buffer_r: buffer da recompensa         
                        List<double> discounted_r = new List<double>(); // Cria um array pra armazenar as recompensas calculadas
                        double ep_r = 0;    // Recompensa do episódio
                        //  Loop de episódio //
                        for (int eplen = 0; eplen < EP_LEN; eplen++)    // Duração de cada episodio
                        {
                            env.render();   // Renderiza o ambiente
                            var a = ppo.choose_action(s);   // Envia um estado s e recebe uma açao a 
                            dynamic step = env.step(a);     // Envia uma açao a ao ambiente e recebe o estado s_, e a recompensa r
                            PyObject s_ = step.GetItem(0);  // Adiciona ao buffer de estado o estado atual s
                            dynamic r = step.GetItem(1);
                            buffer_s.Add(s);    // Adiciona ao buffer de estado o estado atual s
                            buffer_a.Add(a);    // Adiciona ao buffer de ação a açao atual a
                            double _r = r;
                            buffer_r.Add((_r + 8) / 8); // Adiciona ao buffer de recompensa a recompensa atual (?) normalizada (r+8)/8
                            s = s_;     // Atualiza a variavel de estado com o estado recebido pelo ambiente
                            ep_r += _r; // soma a recompensa da ação a recompensa do episodio

                            //  Atualiza PPO //
                            if ((eplen + 1) % BATCH == 0 || eplen == EP_LEN - 1)
                            {
                                double v_s_ = ppo.get_v(s_);    // Passa o estado atual s_ e recebe o valor atual da taxa de aprendizagem da CRITICA
                                                                // V = learned state-value function
                                discounted_r.Clear();           // Limpa o array de recompensas calculadas
                                buffer_r.Reverse();             // Coloca o array de buffer de recompensa ao contrario
                                foreach (double b_r in buffer_r)
                                {
                                    v_s_ = b_r + GAMMA * v_s_;  // Calcula a recompensa multiplicando a recompensa recebida r pela GAMMA 
                                                                // e pelo valor da taxa de aprendizado do estado v_s_
                                    discounted_r.Add(v_s_);     // Adiciona ao array de recompensas calculadas 
                                }
                                discounted_r.Reverse();         // Coloca o array de recompensas calculadas ao contrario
                                // vstack transforma os arrays que estão em linha, em colunas
                                // Esses arrays de colunas sao armazenados em bs ba e br
                                var bs = np.vstack(buffer_s);
                                var ba = np.vstack(buffer_a);
                                var _br = np.array(discounted_r);
                                var br = np.expand_dims(_br, axis: 1);
                                // Esvazia os buffers de estado, açao e recompensa
                                buffer_s.Clear();
                                buffer_a.Clear();
                                buffer_r.Clear();
                                // Treine o cliente e o ator (status, ações, desconto de r)
                                ppo.update( // Atualiza as redes com:
                                    bs,     //   Os estados aculmulados
                                    ba,     //   As ações aculmuladas
                                    br      //   As recompensas aculmuladas
                                );
                            }
                        }
                        // Adiciona a recompensa do episodio atual ao array de recompensas
                        if (epmax == 0)
                        {
                            all_ep_r.Add(ep_r);
                        }
                        else
                        {
                            all_ep_r.Add(all_ep_r.Last() * GAMMA + ep_r * 0.1);
                        }
                        // Escreve na tela
                        Console.WriteLine(
                            "Ep:    " + epmax +      // Numero do episodio
                            "   |   Ep_r:   " + ep_r // Recompensa do episodio
                        );
                    }
                    plt.plot( // Plota o grafico de todas as recompensas
                        np.arange(
                            all_ep_r.Count
                        ),
                        all_ep_r
                    );
                    plt.xlabel("Episode");
                    plt.ylabel("Moving averaged episode reward");
                    plt.show();
                }
                //catch (PythonException error)
                catch (Exception error)
                {
                    // Communicate errors with exceptions from within python script -
                    // this works very nice with pythonnet.
                    Console.WriteLine("Error occured: ", error.Message);
                }
            }
        }
    }
}
