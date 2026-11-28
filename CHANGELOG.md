# Changelog / Cambios

Each version is described in English first, then in Spanish. The release workflow copies the
section of the tagged version into the GitHub Release notes.

Cada versión se describe primero en inglés y luego en español. El workflow de release copia la
sección de la versión etiquetada en las notas de la Release de GitHub.

## Unreleased

## 1.3.0-beta

### English

- **Overlay shows the game's real FPS.** The HUD and OSD counted swapchain presents, inflated by frame repetition (a
  60 FPS cap on the 120 Hz panel presents each frame twice) and HyperFG synthetic frames, so a 60 FPS cap looked like
  ~120 and HyperFG on/off looked the same. They now count the game's own `Present` calls; with HyperFG a second line
  shows the presented rate.
- **HyperFG: hysteresis bypass.** When the real frame rate stays slower than the threshold, frame generation turns fully
  off (no history copy, no compute), so HyperFG never costs more than leaving it off. It comes back below 85 % of the
  threshold.
- **Direct3D 9 (DmC): graphical glitches fixed.** Static vertex/index buffer uploads no longer break the Vulkan render
  pass mid-frame; the draw reads from the upload ring and the device-local copy happens outside the pass.
- **32-bit games: DmC now reaches gameplay.** Windows-correct mutexes (owner + recursion; Bink deadlocked on the opening
  credits), `TlsFree` clears the slot for every thread (PhysX reused stale TLS objects), `ExitThread` really ends 32-bit
  threads, low (below 4 GB) memory gaps are found from `/proc/self/maps` and small Direct3D 9 allocations share pooled
  blocks (DmC ran out of low memory while loading a level), and a thread exiting no longer deadlocks JIT range updates.
  First speed-ups: per-function cached call data and direct native calls from translated 32-bit code (IAT and COM
  vtable calls). Still slow (about 4-15 FPS).
- **Overlay: FPS colours show playability, not distance to the cap.** Green from 55 FPS, amber from 30, red only below 30
  (with a low cap such as 30 FPS, green near the cap). A game at 90 FPS with a 120 FPS cap is no longer shown in red.
- **Branding: the game start screen said "HYPERPLAY".** It now shows the Exedra64 name like the rest of the app.
- **32-bit games: controller input reaches the game (DmC opens its main menu).** `GetFocus` now returns the game window for
  32-bit games; Unreal 3 only reads the keyboard and controller while its window has the focus.
- **Direct3D 9 renderer (Vulkan) for 32-bit games: DmC shows its title screen at about 60 FPS.** New `gfx/D3D9GpuRenderer`:
  D3D9 textures (DXT/BC, 8/16-bit, float and depth formats with the D3D9 channel order), render targets with depth/stencil,
  vertex declarations, instancing, blend/depth/stencil/cull/scissor states, sampler states, `Clear`, `StretchRect`,
  `ColorFill`, `UpdateTexture`/`UpdateSurface` and `Present` through the existing Vulkan swapchain. Shaders use the
  new SM3 translator. Full D3D9 default render states.
- **Direct3D 9 shader translator (Shader Model 3 to SPIR-V).** First piece of the Direct3D 9 renderer for 32-bit games:
  `vs_3_0`/`ps_3_0` bytecode is translated to Vulkan SPIR-V (arithmetic, texture sampling, structured if/loop/break,
  emulated alpha test). All 40 DmC shaders seen so far translate and pass `spirv-val`. Shader bytecode with comment
  blocks is now copied correctly.
- **32-bit games: DmC now runs its game loop (60 presents per second, no crashes).** The JIT supports `XLAT`
  (the render thread died in msvcr100's x87 code). New `GetProcessMemoryInfo` (Unreal 3 calls it every frame; the
  arity-less stub shifted the stack and tripped a security-cookie check), `GetComputerName`, `SetThreadIdealProcessor`,
  `SetThreadAffinityMask` and IMM32 without an IME. Unimplemented system functions now pop their exact stdcall
  arguments. 32-bit critical sections keep `OwningThread`/`RecursionCount` and log the owner after 10 s of waiting.
- **JIT: `MOVS`/`STOS` now honor the direction flag (`STD`).** Repeated string copies and fills always moved forward,
  so an overlapping `memmove` (MSVC copies backwards with `STD; REP MOVSD`) repeated the first element and wrote
  past the end of the buffer. In DmC this corrupted Wwise's memory pool right after the Direct3D 9 device was created.
  DmC now runs past that point and presents its first frame. New unit test; 64-bit games unchanged.
- **32-bit games: pointer-size fixes in Windows synchronization and thread data.** `SLIST` operations used the
  64-bit layout (they wrote 8 bytes into each 4-byte entry and past the 8-byte header), `InitializeSRWLock` wrote 8 bytes,
  TLS expansion slots stored a 64-bit pointer in the 32-bit TEB, and static TLS blocks of new threads were stored as
  64-bit pointers in a table of 4-byte entries. All now use the x86 layout. The 64-bit path is unchanged.
- **32-bit games: Direct3D 9 device (stage 1, no drawing yet).** `IDirect3DDevice9` with all 119 methods and
  its resources (textures, cube/volume textures, surfaces, vertex/index buffers, shaders, declarations, queries,
  swap chain) in low memory with real `Lock`/`Unlock` memory; `Present` keeps a 60 Hz pace. DmC now creates
  its D3D9 device. Also for 32-bit games: window-message functions with 4-byte `WPARAM`/`LPARAM` (`PostMessageW`
  shifted the stack), a real `ShowCursor` counter, a per-thread message queue (`PostThreadMessage`/`GetMessage`
  no longer spin), PIX/cursor stubs with exact signatures. New diagnostics: write watch (`debug.hyperplay.wwatch`)
  and unmap watch (`debug.hyperplay.uwatch`).
- **32-bit games: audio and input foundations (DmC).** New DirectSound 8 over AAudio and 32-bit proxies for
  the WASAPI device objects (Wwise in DmC), and a minimal DirectInput 8 (keyboard/mouse devices without data;
  controllers keep using XInput). Fixed `CoCreateInstance` clearing 8 bytes of a 32-bit out-pointer (it
  overwrote the caller's saved frame pointer). DmC now passes audio and input setup; it needs a Direct3D 9
  renderer next (its Direct3D 11 path is disabled in the shipped executable).
- **DmC: the Unreal Engine 3 game now loads its packages and starts the engine (window, splash, shaders,
  PhysX, game packages).** Unreal 3 games (32-bit) now see the real install layout: the .exe is presented
  as `D:\Binaries\Win32\` and `D:\` is the install root, so `..\..\` paths collapse like on Windows
  (before, packages were read from a wrong path and reported as "unrecognizable data"). The game is started
  with `-dx11` (Direct3D 11 renderer, shipped SM5 shaders). Added: a query-only `IDirect3D9` (GPU
  identification), `SetCurrentDirectory`/`GetCurrentDirectory` for 32-bit games, VC++ 2008 runtime calls
  served by the VC++ 2010 runtime, executable heaps (.NET stubs), WinINet without network, and exact
  signatures for `CallNtPowerInformation`, `D3DXCheckVersion`, `ReadProcessMemory`, console and
  debug-output functions. Not done yet: audio (Wwise needs WASAPI for 32-bit) and the Direct3D 11 renderer
  for 32-bit games.
- **32-bit games (DmC): Steam is emulated for 32-bit games** (`steam_api.dll`: Apps, User, Friends,
  Utils, UserStats, RemoteStorage saved in `_HyperPlaySaves/steam_cloud`). The 64-bit Steam emulation is
  unchanged. Added the x86 `EMMS` instruction and `FileTimeToLocalFileTime`/`LocalFileTimeToFileTime`.
- **Safety: games can no longer delete, move or replace files outside their own folder.** Paths reached
  with `..\` (the Unreal install root) are read and create only; delete, rename and replace return
  "access denied".
- **DmC: the .NET runtime now starts completely and the Unreal engine runs** until Steam initialization.
  Added: hardware exceptions for 32-bit games (vectored handlers + `fs:[0]` chain), WinRT IIDs
  (`RoGetParameterizedTypeInstanceIID`), Winsock without network, XInput/Winsock/OLEAUT32 imports by
  ordinal, PE resources (`FindResource`...), ordinal-only exports, `..` paths up to the game root,
  file URLs (`UrlCanonicalizeW`), and the x86 instructions `SAHF`/`LAHF`. Fixed `PUSH ESP`/`POP ESP`,
  `WaitForMultipleObjects` and `TerminateProcess` in 32-bit games.
- **32-bit games (DmC): the .NET Framework 4 runtime now starts.** Windows x86 exception
  handling (the `fs:[0]` frame chain, `RtlUnwind`, `NtContinue`) is implemented, so C++ `catch`
  blocks work in 32-bit games. 32-bit memory (`VirtualAlloc`, `MapViewOfFile`) stays below 4 GB,
  pointer-sized out-parameters are written as 4 bytes, 64-bit arguments take two stack slots, and
  many missing Windows functions were added (security/SID, OLEAUT32 by ordinal, URL/path helpers,
  memory status, directories, `PUSHAD`/`POPAD`, `RDTSC`). DmC still stops while loading `mscorlib`.
- Fixed: the window-message log wrote every message (a 400 MB log in Celeste); now only the first 50.

### Español

- **El overlay muestra los FPS reales del juego.** El HUD y el OSD contaban las presentaciones del swapchain, infladas
  por la repetición de frames (un tope de 60 FPS en el panel de 120 Hz presenta cada frame dos veces) y por los frames
  sintéticos de HyperFG: con tope 60 se veían ~120 y HyperFG encendido o apagado se veía igual. Ahora cuentan las
  llamadas `Present` del propio juego; con HyperFG una segunda línea muestra lo presentado.
- **HyperFG: bypass con histéresis.** Si el ritmo real se mantiene más lento que el umbral, la generación se apaga del
  todo (sin copia al historial ni cómputo), así que HyperFG nunca cuesta más que tenerlo apagado. Vuelve bajo el 85 %
  del umbral.
- **Direct3D 9 (DmC): errores gráficos corregidos.** Las subidas de búferes de vértices/índices estáticos ya no cortan el
  render pass de Vulkan a mitad de frame; el draw lee del anillo de subida y la copia a memoria de GPU se hace fuera del
  pase.
- **Juegos de 32 bits: DmC ya llega a la partida.** Mutex como en Windows (dueño y recursión; Bink se bloqueaba en los créditos
  iniciales), `TlsFree` vacía la ranura en todos los hilos (PhysX reutilizaba objetos TLS viejos), `ExitThread` termina de verdad
  los hilos de 32 bits, los huecos de memoria por debajo de 4 GB se buscan en `/proc/self/maps` y las reservas pequeñas de
  Direct3D 9 comparten bloques (DmC se quedaba sin memoria baja al cargar un nivel), y un hilo que termina ya no bloquea al
  JIT. Primeras mejoras de velocidad: datos de llamada en caché por función y llamadas nativas directas desde el código
  traducido de 32 bits (IAT y vtables COM). Aún va lento (unos 4-15 FPS).
- **Overlay: los colores de FPS indican si el juego es jugable, no lo lejos que está del tope.** Verde desde 55 FPS, ámbar
  desde 30 y rojo solo por debajo de 30 (con un tope bajo, como 30 FPS, verde cerca del tope). Un juego a 90 FPS con tope 120
  ya no sale en rojo.
- **Marca: la pantalla de inicio de un juego decía "HYPERPLAY".** Ahora muestra el nombre Exedra64 como el resto de la app.
- **Juegos de 32 bits: el mando llega al juego (DmC abre su menú principal).** `GetFocus` devuelve ahora la ventana del
  juego en 32 bits; Unreal 3 solo lee teclado y mando mientras su ventana tiene el foco.
- **Renderizador Direct3D 9 (Vulkan) para juegos de 32 bits: DmC muestra su pantalla de título a unos 60 FPS.** Nuevo
  `gfx/D3D9GpuRenderer`: texturas D3D9 (DXT/BC, formatos de 8/16 bits, flotantes y de profundidad con el orden de canales de
  D3D9), render targets con profundidad/stencil, declaraciones de vértices, instancias, estados de mezcla/profundidad/stencil/
  culling/tijera, estados de sampler, `Clear`, `StretchRect`, `ColorFill`, `UpdateTexture`/`UpdateSurface` y `Present`
  por el swapchain Vulkan existente. Los shaders usan el nuevo traductor SM3. Estados por defecto de D3D9 completos.
- **Traductor de shaders Direct3D 9 (Shader Model 3 a SPIR-V).** Primera pieza del renderizador Direct3D 9 para juegos de
  32 bits: el bytecode `vs_3_0`/`ps_3_0` se traduce a SPIR-V de Vulkan (aritmética, muestreo de texturas, if/bucles/break
  estructurados, prueba alfa emulada). Los 40 shaders de DmC vistos hasta ahora se traducen y pasan `spirv-val`. El
  bytecode con bloques de comentario se copia ahora bien.
- **Juegos de 32 bits: DmC ya ejecuta su bucle de juego (60 Present por segundo, sin fallos).** El JIT admite `XLAT`
  (el hilo de render moría en el código x87 de msvcr100). Nuevos `GetProcessMemoryInfo` (Unreal 3 lo llama cada
  fotograma; el stub sin aridad descuadraba la pila y saltaba la cookie de seguridad), `GetComputerName`,
  `SetThreadIdealProcessor`, `SetThreadAffinityMask` e IMM32 sin IME. Las funciones de sistema sin implementar sacan
  ahora sus argumentos stdcall exactos. Las secciones críticas de 32 bits mantienen `OwningThread`/`RecursionCount` y
  avisan del dueño tras 10 s de espera.
- **JIT: `MOVS`/`STOS` respetan ahora el flag de dirección (`STD`).** Las copias y rellenos repetidos siempre avanzaban,
  así que un `memmove` solapado (MSVC copia hacia atrás con `STD; REP MOVSD`) repetía el primer elemento y escribía
  más allá del final del búfer. En DmC esto corrompía el montón de Wwise justo después de crear el dispositivo Direct3D 9.
  DmC ya pasa ese punto y presenta su primer fotograma. Prueba unitaria nueva; los juegos de 64 bits no cambian.
- **Juegos de 32 bits: tamaño de puntero en la sincronización de Windows y los datos por hilo.** Las `SLIST` usaban el
  formato de 64 bits (escribían 8 bytes en cada entrada de 4 y más allá de la cabecera de 8), `InitializeSRWLock`
  escribía 8 bytes, las ranuras TLS de expansión guardaban un puntero de 64 bits en el TEB de 32 bits y los bloques TLS
  estáticos de los hilos nuevos se guardaban como punteros de 64 bits en una tabla de entradas de 4 bytes. Todo usa
  ahora el formato x86. La ruta de 64 bits no cambia.
- **Juegos de 32 bits: dispositivo Direct3D 9 (etapa 1, aún sin dibujar).** `IDirect3DDevice9` con sus 119 métodos
  y sus recursos (texturas, cubos/volúmenes, superficies, búferes de vértices/índices, shaders, declaraciones,
  consultas, cadena de intercambio) en memoria baja con memoria real en `Lock`/`Unlock`; `Present` mantiene
  60 Hz. DmC ya crea su dispositivo D3D9. Además, en 32 bits: funciones de mensajes de ventana con `WPARAM`/`LPARAM`
  de 4 bytes (`PostMessageW` desplazaba la pila), contador real de `ShowCursor`, cola de mensajes por hilo
  (`PostThreadMessage`/`GetMessage` ya no giran en vacío) y stubs de PIX/cursor con firma exacta. Nuevos
  diagnósticos: vigía de escrituras (`debug.hyperplay.wwatch`) y de liberaciones (`debug.hyperplay.uwatch`).
- **Juegos de 32 bits: base de audio y entrada (DmC).** Nuevo DirectSound 8 sobre AAudio y proxies de 32 bits
  para los objetos de dispositivo WASAPI (Wwise en DmC), y un DirectInput 8 mínimo (teclado/ratón sin datos;
  los mandos siguen por XInput). Corregido `CoCreateInstance`, que borraba 8 bytes de un puntero de salida de
  32 bits (pisaba el puntero de marco guardado del llamador). DmC ya pasa la inicialización de audio y entrada;
  ahora necesita un renderizador Direct3D 9 (su ruta Direct3D 11 está desactivada en el ejecutable).
- **DmC: el juego de Unreal Engine 3 ya carga sus paquetes y arranca el motor (ventana, splash, shaders,
  PhysX, paquetes del juego).** Los juegos de Unreal 3 (32 bits) ven ahora la estructura real de la instalación:
  el .exe aparece como `D:\Binaries\Win32\` y `D:\` es la raíz, así las rutas `..\..\` se resuelven como en
  Windows (antes los paquetes se leían de una ruta equivocada y daban "unrecognizable data"). El juego arranca
  con `-dx11` (renderizador Direct3D 11, shaders SM5 incluidos). Añadido: `IDirect3D9` de solo consulta
  (identificar la GPU), `SetCurrentDirectory`/`GetCurrentDirectory` en 32 bits, llamadas del runtime VC++ 2008
  atendidas por el de VC++ 2010, montones ejecutables (stubs de .NET), WinINet sin red y firmas exactas de
  `CallNtPowerInformation`, `D3DXCheckVersion`, `ReadProcessMemory` y funciones de consola y depuración.
  Falta: audio (Wwise necesita WASAPI de 32 bits) y el renderizador Direct3D 11 para juegos de 32 bits.
- **Juegos de 32 bits (DmC): Steam se emula en juegos de 32 bits** (`steam_api.dll`: Apps, User, Friends,
  Utils, UserStats y RemoteStorage guardado en `_HyperPlaySaves/steam_cloud`). La emulación de Steam de 64
  bits no cambia. Añadidas la instrucción x86 `EMMS` y `FileTimeToLocalFileTime`/`LocalFileTimeToFileTime`.
- **Seguridad: los juegos ya no pueden borrar, mover ni reemplazar archivos fuera de su carpeta.** Las rutas
  con `..\` (la raíz de la instalación de Unreal) son de solo lectura y creación; borrar, renombrar y
  reemplazar devuelven "acceso denegado".
- **DmC: el runtime de .NET ya arranca por completo y el motor Unreal se ejecuta** hasta iniciar
  Steam. Añadido: excepciones de hardware en juegos de 32 bits (manejadores vectoriales + cadena
  `fs:[0]`), IID de WinRT (`RoGetParameterizedTypeInstanceIID`), Winsock sin red, importaciones por
  ordinal de XInput/Winsock/OLEAUT32, recursos PE (`FindResource`...), exportaciones solo por ordinal,
  rutas con `..` hasta la raíz del juego, URL de archivo (`UrlCanonicalizeW`) y las instrucciones x86
  `SAHF`/`LAHF`. Corregidos `PUSH ESP`/`POP ESP`, `WaitForMultipleObjects` y `TerminateProcess` en
  juegos de 32 bits.
- **Juegos de 32 bits (DmC): el runtime de .NET Framework 4 ya arranca.** Se implementa el manejo
  de excepciones de Windows x86 (cadena de marcos en `fs:[0]`, `RtlUnwind`, `NtContinue`): los
  `catch` de C++ funcionan en juegos de 32 bits. La memoria de 32 bits (`VirtualAlloc`,
  `MapViewOfFile`) queda por debajo de 4 GB, los parámetros de salida de tamaño puntero se escriben
  con 4 bytes, los argumentos de 64 bits ocupan dos ranuras de pila y se añaden muchas funciones de
  Windows que faltaban (seguridad/SID, OLEAUT32 por ordinal, URL y rutas, estado de memoria,
  carpetas, `PUSHAD`/`POPAD`, `RDTSC`). DmC aún se para al cargar `mscorlib`.
- Corregido: el registro de mensajes de ventana escribía cada mensaje (log de 400 MB en Celeste);
  ahora solo los 50 primeros.
## 1.2.0

### English

- **New name: HyperPlay is now Exedra64.** New name, icon and logo (chrome gamepad with "E64":
  the E doubles as the D-pad, with Select/Start and L1/R1). The GitHub repository moved to
  `murdoc820/Exedra64` (old links redirect). Installs, saves and updates are unaffected: the
  package, the `/sdcard/HyperPlay/` folder and `_HyperPlaySaves` keep their names. The release APK
  is now `Exedra64-vX.Y.Z-Snapdragon8Elite-arm64-v8a.apk`; the updater accepts both names.
- **Stardew Valley is playable (in testing):** controller input works (window focus and cursor
  fixed), starting a new game no longer crashes (I/O completion ports, thread pool), the audio is
  no longer distorted (a limiter replaces hard clipping), and characters draw complete
  (`glGetTexImage` implemented).
- **32-bit games (groundwork for DmC):** 32-bit executables now get memory, stack and TEB below
  4 GB, a stdcall/cdecl bridge to native functions, 32-bit layouts of Windows structures
  (`CRITICAL_SECTION`, `SYSTEM_INFO`, `MEMORY_BASIC_INFORMATION`, `STARTUPINFO`, SRW locks) and
  up to 65536 DLL exports. The .NET Framework 4.8 from Microsoft can be used by 32-bit games
  (registry keys, `C:\Windows` mapping, `mscoree`). DmC still stops at startup.
- Engine: `PCMPISTRI` fixed (invalid-element rules).
- **Fixed: in-app updates now install.** The update downloaded but the system "Install?" dialog
  never appeared, so it stayed on "Installing…" forever: it was opened from a broadcast receiver,
  which Android 15/16 silently blocks. The launcher screen now opens it. Install failures show a
  clear reason (cancelled, blocked, signature/version conflict, invalid APK, no space), GitHub
  errors are no longer reported as "up to date", and the release APK is picked by its official name.
- **Privacy: error reports are scrubbed before they leave the phone.** IP and MAC addresses,
  emails, user folders (`/storage/emulated/0/...`, `/sdcard/...`, `C:\Users\<name>`), device
  serials, Android ID, IMEI-like numbers, SteamIDs, account names and tokens are replaced with
  placeholders; of storage paths only the game folder and file name are kept. The report server
  applies the same rules again and can send reports to a separate private repository
  (`REPORTS_REPO`).
- **Engine: real Windows x64 exception handling (SEH).** C++ and .NET exceptions are now
  dispatched like on Windows: vectored handlers, then each frame's handler from the module's
  `.pdata` (and the JIT tables of Mono/CoreCLR), then the unhandled filter; `RtlUnwindEx`,
  `RtlRestoreContext` (C++ catch blocks, `longjmp`), `RtlVirtualUnwind`, `RtlLookupFunctionEntry`
  and `RtlCaptureContext` are implemented. Before, `RaiseException` did nothing.
- **Stardew Valley (work in progress):** its .NET 6 runtime (CoreCLR) now starts, loads the game
  and MonoGame and initializes SDL. Fixed on the way: forwarded exports of local DLLs, `SEC_IMAGE`
  file mappings, 64 KB-aligned `VirtualAlloc`, unaligned `VirtualProtect`, `GetCPInfo`, CET
  instructions (`RDSSP`), Windows version checks, file functions used by saves (`SetEndOfFile`,
  `SetFileInformationByHandle`, real `FlushFileBuffers`). Not yet: the game draws with OpenGL,
  which HyperPlay does not have yet.
- **OpenGL for Windows games (first version):** HyperPlay now has its own `opengl32.dll`: WGL
  contexts are EGL contexts and desktop OpenGL calls go to the phone's OpenGL ES 3.2 (float
  arguments read from XMM registers, desktop-only calls adapted, GLSL `#version` lines removed),
  plus the GDI that SDL needs (pixel formats, `SwapBuffers`, DCs, cursor bitmaps). Stardew Valley's
  SDL video now starts. Not verified yet: no OpenGL game draws on screen yet.
- Engine: `PUSHFQ`/`POPFQ`, `NtQuerySystemInformation` (process list), .NET settings now passed in
  the environment block that CoreCLR reads, SDL uses only XInput controllers in every game, safer
  exception dispatch over corrupt stacks.
- Fixed: an old Unity-only workaround wrote into whatever a game stored in TLS slot 1; in
  Stardew Valley it corrupted .NET objects. It now applies only to Unity games.
- Exceptions: `RtlVirtualUnwind` now reports where registers were saved (the .NET garbage
  collector needs it), and resuming after a .NET `catch` (CoreCLR jumps to the target frame by
  itself) is detected and applied correctly. Translator: `PTEST`.
- Stardew Valley now gets through `Game1`'s startup and stops later (`LocalMultiplayer`, a type
  that fails to load); under investigation.
- Tested: Celeste (60 FPS) and Hollow Knight (menu) still work.
- Engine: `FlushInstructionCache` now discards translated code for that range (it was a stub that
  did nothing and returned failure). New diagnostics for .NET games: exception codes and runtime
  exception texts in the log, and an optional EventPipe trace (`debug.hyperplay.dotnettrace 1`).
  Stardew Valley still stops at startup (a type lookup by name fails inside XmlSerializer).
- **.NET runtime messages now show:** `LoadStringW` reads string resources from PE files (CoreCLR
  takes its error texts from `mscorrc.dll`) and `GetUserDefaultLocaleName` returns `en-US` (it was a
  stub returning failure). Before, every .NET runtime error had an empty message.
- Engine: the safe string functions of the C runtime (`strcpy_s`, `strncpy_s`, `strcat_s`,
  `wcsncpy_s`...) and the static `memcpy` inside native DLLs such as `coreclr.dll` and
  `UnityPlayer.dll` now run as native ARM code instead of translated x86 (exact Windows behavior,
  faster). .NET games run without the concurrent background garbage collector.
- Tested: Celeste, Cuphead, Hollow Knight and Silksong still at 120 FPS.
- **Display settings now live in each game.** The quick menu no longer has Screen (Stretch / Zoom /
  aspect) or the render-resolution percentage: games always run at the screen's native resolution,
  full screen and undistorted. The game's own video menu now lists real resolutions (720p, 900p,
  1080p, 1/2, 5/8 and 3/4 of the screen, and native) through `EnumDisplaySettings` and DXGI
  `GetDisplayModeList`; a lower resolution is scaled to the screen keeping its aspect ratio (black
  bars when it differs), also for games with their own Vulkan (Celeste). Tested in Hollow Knight:
  2016x912 fills the screen and the GPU load drops from 75 % to 57 %.
- **FPS cap: 60, 90 or 120** (72 and Unlimited removed; old choices move to the default).
- **Controller navigation in the quick menu:** left/right change the focused option and never move
  to another row; from the action buttons, right jumps to the settings column; LB/RB switch
  columns; the D-pad no longer counts twice (key + hat) and the stick uses only its main direction.
  The upscaler's automatic choice is now just called "Automatic".
- **Launcher with a controller:** "up" from the game covers no longer leaves the focus on the
  Home/Settings tabs, where it was invisible (tabs switch with LB/RB); the same for "back" on the
  game page (B). The update notice shows a frame when focused and opens with A.
- Also tested: Silksong and Cuphead at a lower resolution (16:9 with black bars or full screen,
  undistorted) and Silksong's own frame limit (30 real FPS, doubled to 60 by HyperFG).

### Español

- **Nuevo nombre: HyperPlay ahora es Exedra64.** Nombre, icono y logo nuevos (mando cromado con
  "E64": la E hace de cruceta, con Select/Start y L1/R1). El repositorio de GitHub pasa a
  `murdoc820/Exedra64` (los enlaces antiguos redirigen). Instalaciones, partidas y actualizaciones
  no cambian: el paquete, la carpeta `/sdcard/HyperPlay/` y `_HyperPlaySaves` conservan su nombre.
  El APK de cada versión se llama ahora `Exedra64-vX.Y.Z-Snapdragon8Elite-arm64-v8a.apk`; el
  actualizador acepta los dos nombres.
- **Stardew Valley es jugable (en pruebas):** el mando funciona (foco de ventana y cursor
  corregidos), empezar partida nueva ya no se cierra (puertos de finalización de E/S, pool de
  hilos), el sonido ya no se distorsiona (un limitador sustituye el recorte brusco) y los
  personajes se dibujan completos (`glGetTexImage` implementado).
- **Juegos de 32 bits (base para DmC):** los ejecutables de 32 bits reciben memoria, pila y TEB por
  debajo de 4 GB, un puente stdcall/cdecl a las funciones nativas, las estructuras de Windows con
  su forma de 32 bits (`CRITICAL_SECTION`, `SYSTEM_INFO`, `MEMORY_BASIC_INFORMATION`,
  `STARTUPINFO`, bloqueos SRW) y hasta 65536 exportaciones por DLL. Los juegos de 32 bits pueden
  usar el .NET Framework 4.8 de Microsoft (claves de registro, carpeta `C:\Windows`, `mscoree`).
  DmC aún se para al arrancar.
- Motor: corregido `PCMPISTRI` (reglas de elementos no válidos).
- **Arreglado: las actualizaciones desde la app ya se instalan.** La actualización se descargaba,
  pero el diálogo «¿Instalar?» del sistema no aparecía y se quedaba en «Instalando…» para siempre:
  se abría desde un receptor de broadcasts, algo que Android 15/16 bloquea sin avisar. Ahora lo
  abre la pantalla del launcher. Los fallos de instalación muestran un motivo claro (cancelada,
  bloqueada, conflicto de firma/versión, APK no válido, sin espacio), los errores de GitHub ya no
  se muestran como «al día» y el APK de la Release se elige por su nombre oficial.
- **Privacidad: los informes de errores se limpian antes de salir del teléfono.** Direcciones IP
  y MAC, correos, carpetas de usuario (`/storage/emulated/0/...`, `/sdcard/...`,
  `C:\Users\<nombre>`), números de serie, Android ID, números tipo IMEI, SteamID, nombres de cuenta
  y tokens se sustituyen por marcadores; de las rutas de almacenamiento solo se conservan la
  carpeta del juego y el nombre del archivo. El servidor de informes vuelve a aplicar las mismas
  reglas y puede enviar los informes a un repositorio privado aparte (`REPORTS_REPO`).
- **Motor: manejo real de excepciones de Windows x64 (SEH).** Las excepciones de C++ y .NET se
  despachan como en Windows: manejadores vectoriales, después el manejador de cada marco según el
  `.pdata` del módulo (y las tablas del JIT de Mono/CoreCLR), y por último el filtro de no
  manejadas. Implementados `RtlUnwindEx`, `RtlRestoreContext` (bloques catch de C++, `longjmp`),
  `RtlVirtualUnwind`, `RtlLookupFunctionEntry` y `RtlCaptureContext`. Antes `RaiseException` no
  hacía nada.
- **Stardew Valley (en curso):** su runtime .NET 6 (CoreCLR) ya arranca, carga el juego y
  MonoGame e inicia SDL. Arreglado por el camino: exports reenviados de DLL locales, mapeos
  `SEC_IMAGE`, `VirtualAlloc` alineado a 64 KB, `VirtualProtect` no alineado, `GetCPInfo`,
  instrucciones CET (`RDSSP`), comprobaciones de versión de Windows y funciones de archivo que usan
  los guardados (`SetEndOfFile`, `SetFileInformationByHandle`, `FlushFileBuffers` real). Falta: el
  juego dibuja con OpenGL, que HyperPlay aún no tiene.
- **OpenGL para juegos de Windows (primera versión):** HyperPlay tiene su propio `opengl32.dll`:
  los contextos WGL son contextos EGL y las llamadas de OpenGL de escritorio van al OpenGL ES 3.2
  del teléfono (argumentos float leídos de los registros XMM, llamadas solo de escritorio
  adaptadas, líneas `#version` de GLSL quitadas), más el GDI que necesita SDL (formatos de píxel,
  `SwapBuffers`, DC, mapas de bits de cursor). El vídeo SDL de Stardew Valley ya arranca. Sin
  verificar aún: ningún juego OpenGL dibuja todavía en pantalla.
- Motor: `PUSHFQ`/`POPFQ`, `NtQuerySystemInformation` (lista de procesos), los ajustes de .NET van
  ahora en el bloque de entorno que lee CoreCLR, SDL usa solo mandos XInput en todos los juegos y
  el despacho de excepciones resiste pilas corruptas.
- Arreglado: un apaño antiguo solo para Unity escribía en lo que cualquier juego guardara en el
  slot TLS 1; en Stardew Valley corrompía objetos .NET. Ahora solo se aplica a juegos Unity.
- Excepciones: `RtlVirtualUnwind` indica ahora dónde se guardó cada registro (lo necesita el
  recolector de basura de .NET), y la reanudación tras un `catch` de .NET (CoreCLR salta él mismo
  al marco destino) se detecta y se aplica bien. Traductor: `PTEST`.
- Stardew Valley ya pasa el arranque de `Game1` y se para más adelante (`LocalMultiplayer`, un
  tipo que no carga); en investigación.
- Probado: Celeste (60 FPS) y Hollow Knight (menú) siguen funcionando.
- Motor: `FlushInstructionCache` descarta ahora el código traducido de ese rango (era un stub que
  no hacía nada y devolvía fallo). Diagnósticos nuevos para juegos .NET: códigos y textos de las
  excepciones del runtime en el registro, y una traza EventPipe opcional
  (`debug.hyperplay.dotnettrace 1`). Stardew Valley sigue parándose al arrancar (falla la búsqueda de
  un tipo por nombre dentro de XmlSerializer).
- **Los mensajes del runtime de .NET ya se ven:** `LoadStringW` lee los recursos de texto de los
  archivos PE (CoreCLR saca sus textos de error de `mscorrc.dll`) y `GetUserDefaultLocaleName`
  devuelve `en-US` (era un stub que devolvía fallo). Antes todos los errores del runtime de .NET
  salían sin mensaje.
- Motor: las funciones seguras de cadenas del CRT (`strcpy_s`, `strncpy_s`, `strcat_s`,
  `wcsncpy_s`...) y el `memcpy` estático de DLL nativas como `coreclr.dll` y `UnityPlayer.dll` se
  ejecutan ahora como código ARM nativo en vez de x86 traducido (comportamiento exacto de Windows y
  más rápido). Los juegos .NET van sin el recolector de basura concurrente en segundo plano.
- Probado: Celeste, Cuphead, Hollow Knight y Silksong siguen a 120 FPS.
- **Los ajustes de pantalla están ahora en cada juego.** El menú rápido ya no tiene Pantalla
  (Estirar / Ampliar / aspecto) ni la resolución en porcentaje: los juegos van siempre a la
  resolución nativa de la pantalla, a pantalla completa y sin deformarse. El menú de vídeo del
  propio juego muestra ahora resoluciones reales (720p, 900p, 1080p, 1/2, 5/8 y 3/4 de la pantalla,
  y la nativa) mediante `EnumDisplaySettings` y `GetDisplayModeList` de DXGI; una resolución menor
  se escala a la pantalla conservando la proporción (bandas negras si es distinta), también en los
  juegos con Vulkan propio (Celeste). Probado en Hollow Knight: 2016x912 llena la pantalla y la
  carga de la GPU baja del 75 % al 57 %.
- **Tope de FPS: 60, 90 o 120** (se quitan 72 e Ilimitado; las elecciones antiguas pasan al valor
  por defecto).
- **Navegación con mando en el menú rápido:** izquierda/derecha cambian la opción enfocada y nunca
  saltan a otra fila; desde los botones de acción, derecha salta a la columna de ajustes; LB/RB
  cambian de columna; la cruceta ya no cuenta doble (tecla + hat) y el stick solo usa su dirección
  principal. La opción automática del reescalado se llama ahora "Automático".
- **Launcher con mando:** "arriba" desde las portadas ya no deja el foco en las pestañas
  Inicio/Ajustes, donde era invisible (se cambian con LB/RB); lo mismo con "volver" en la ficha del
  juego (B). El aviso de actualización muestra un marco al enfocarlo y se abre con A.
- Probado también: Silksong y Cuphead a una resolución menor (16:9 con bandas negras o pantalla
  completa, sin deformar) y el límite de fotogramas del propio Silksong (30 FPS reales, que HyperFG
  duplica a 60).

## 1.1.0

### English

- **Celeste at 120 FPS with HyperFG.** Games that draw with their own Vulkan (Celeste) now draw
  into HyperPlay images; HyperPlay copies each frame to the screen. With HyperFG on (Quick menu),
  HyperPlay generates one frame between each pair of game frames: Celeste runs at 60 and the
  screen shows 120. Tested on the phone: title, main menu and 3D mountain menu at 120 FPS, no
  crashes. Known: HyperFG uses more GPU at native resolution (about 60-80 %), and the thermal
  protection of HyperFG does not act on these games yet.
- The APK name now says the processor it is built for:
  `HyperPlay-vX.Y.Z-Snapdragon8Elite-arm64-v8a.apk`.

- **.NET games (Celeste)**: HyperPlay runs .NET Framework games with Mono x64 for Windows
  (translated) and FNA instead of XNA. The game's Vulkan renderer (FNA3D/SDL GPU) is passed to
  the phone's Vulkan almost as is (Win32 surface → Android surface, orientation handled). Tested
  on the phone: Celeste title, menus, file select and the prologue at 60 FPS with the controller.
  The .NET runtime (33 MB, release `runtime-mono-x64-1`) is downloaded by the app the first time
  a .NET game opens, checked with SHA-256 and installed in the app's own folder. Not yet: audio
  (FMOD is proprietary and the shim is silent; your own FMOD 1.10 x64 for Windows in
  `/sdcard/HyperPlay/fmod` replaces it).
- Engine fixes found on the way: DLL init order (dependencies first), real
  `CommandLineToArgvW`, `GetStringTypeW` (the CRT thought no character was a space),
  `ExitThread`, `ERROR_TIMEOUT` from condition variables, `NtCreateFile`/`NtQueryDirectoryFile`,
  `SetProp`/`GetProp`, `msvcrt.dll` routed to the UCRT, 32-bit DLLs refused in 64-bit games,
  game console output in logcat.
- **Real widescreen for Cuphead** (Quick menu → Image → Screen = **Widescreen**, the default):
  the game draws more of the scene at the sides, at native resolution, with no stretching and
  no cropping. HyperPlay patches the 16:9 limit of the game camera in memory when the game
  loads (the game files are not changed). Tested on the phone: menus, world map and a
  run-and-gun level fill the 20:9 screen at 60 FPS. Known detail: in some levels a haze layer of
  the sky was drawn for 16:9 and shows a faint seam near the old edges.
- Quick menu → Image → **Screen**: Original / Stretch / Zoom, saved per game. Fills the screen
  for games that draw 16:9 with side bars (Cuphead on a 20:9 phone). Stretch widens the image;
  Zoom keeps the shape and crops top and bottom. Tested on the device in Cuphead.
- Each game now runs in its own process. Force closing a game and opening it (or another one)
  again no longer crashes the app. Quick Resume still works (the held game resumes instantly);
  opening a different game closes the held one first.
- Launcher: a D-pad press is no longer lost after the RedMagic Game Space layer shows up when
  the app opens.
- Quick menu: tapping ‹ now steps a setting back and › steps it forward (before, any tap moved
  forward).
- Gamepad: very short button presses are held for at least 40 ms, so games that read the pad
  once per frame no longer miss them.
- **Hollow Knight: Silksong starts without internet.** About 1 launch in 5 stayed on a black
  screen (with or without internet; it looked like a network problem). Cause:
  `TryEnterCriticalSection` waited like `EnterCriticalSection` and the Mono garbage collector,
  which uses it while it pauses threads, waited forever. Tested on the phone: 10 of 10 launches
  without internet reach the menu (~115 FPS), and a new game is playable.
- Fixed a crash when a game opened a file deep in its call stack (Silksong when starting a game):
  the Cuphead widescreen code used 64 KB of stack in every file open. The game thread and the
  threads games create now also get a larger native stack.
- A thread that runs game code from inside a native call (callbacks, exceptions) is marked as
  "in native code" again when it returns, so pausing it never waits forever.
- **Celeste has sound, without FMOD.** HyperPlay now has its own FMOD Studio (native code, no
  Firelight code): it reads the game's sound banks, decodes their Vorbis sounds with Android and
  mixes them at 48 kHz. Tested on the phone: title ambience (looping), menu music and interface
  sounds. Supported: events and timelines, random variations, nested events, loops and sustain
  points, bus/VCA volume and pause, long music decoded in blocks, a soft limiter. Tested in the
  prologue: footsteps, jump, landing, ambience and music. Also: game parameters (music layers fade
  in and out like on PC; defaults read from the bank), sounds chosen by a parameter, and stereo
  position (left/right, quieter off screen), the mixer buses and their faders, snapshots (the
  pause menu lowers the music and ambience; story moments lower the ambience) and music that
  changes section when the game says so. Not done: effects such as reverb. Celeste's saves were
  checked: "Save and Quit" writes the file and it loads
  after restarting.
- Loading screen: games that draw with their own Vulkan (Celeste) now leave it as soon as they
  show something (before it stayed up to 45 s).
- Fixed: Mono sometimes aborted (`mono-mmap-windows.c:132`) because `VirtualQuery` reported a
  just-reserved region as free; and a folder listing could come back empty when a handle number
  was reused (Celeste: "Missing Language Files").
- **Celeste saves again.** Changing an option showed "Save Failed": Mono creates the `Saves`
  folder with a Win32 long path (`\\?\D:\Saves`) and HyperPlay did not understand the `\\?\`
  prefix. Tested on the phone: an option changed in Celeste is still there after restarting.
- Loading screen: the technical engine text is gone; it now shows a friendly stage ("Getting the
  game ready…", "Almost there…") and a light-hearted line that changes every few seconds.
- .NET detection fixed: mixed C++/CLI executables (DmC: Devil May Cry) were started with Mono.
  Only IL-only assemblies use Mono now.
- 32-bit games: their DLLs are always mapped below 4 GB (before, some PhysX DLLs of DmC ended up
  above it), and the import table of a 32-bit DLL is written with 4-byte slots (DmC crashed while
  loading its wxWidgets DLLs). DmC still does not start (missing Windows functions).
- **Error reports reach the server.** With your permission, reports are sent to the HyperPlay
  report server, which opens an issue on GitHub (no token in the app). To use little space on
  the phone: the app log is cleared after reports are sent, at most 20 unsent reports (under 30
  days old) are kept, and only the last 5 game session logs.

### Español

- **Celeste a 120 FPS con HyperFG.** Los juegos que dibujan con su propio Vulkan (Celeste) dibujan
  ahora en imágenes de HyperPlay, y HyperPlay copia cada frame a la pantalla. Con HyperFG activo
  (Menú rápido), HyperPlay genera un frame entre cada par de frames del juego: Celeste va a 60 y
  la pantalla muestra 120. Probado en el teléfono: título, menú principal y menú 3D de la montaña
  a 120 FPS, sin cierres. Conocido: HyperFG usa más GPU a resolución nativa (60-80 %), y la
  protección térmica de HyperFG aún no actúa en estos juegos.
- El nombre del APK indica ahora el procesador para el que está hecho:
  `HyperPlay-vX.Y.Z-Snapdragon8Elite-arm64-v8a.apk`.
- **Juegos .NET (Celeste)**: HyperPlay ejecuta juegos de .NET Framework con Mono x64 para Windows
  (traducido) y FNA en lugar de XNA. El renderizador Vulkan del juego (FNA3D/SDL GPU) se pasa al
  Vulkan del teléfono casi tal cual (superficie Win32 → Android, orientación resuelta). Probado en
  el teléfono: título, menús, selección de partida y el prólogo de Celeste a 60 FPS con mando.
  La app descarga el runtime .NET (33 MB, release `runtime-mono-x64-1`) la primera vez que se abre
  un juego .NET, lo comprueba con SHA-256 y lo instala en su propia carpeta. Falta: sonido (FMOD
  es propietario y el shim es mudo; tu propio FMOD 1.10 x64 para Windows en
  `/sdcard/HyperPlay/fmod` lo sustituye).
- Arreglos del motor encontrados por el camino: orden de inicio de las DLL (dependencias
  primero), `CommandLineToArgvW` real, `GetStringTypeW` (el CRT creía que ningún carácter era un
  espacio), `ExitThread`, `ERROR_TIMEOUT` en variables de condición,
  `NtCreateFile`/`NtQueryDirectoryFile`, `SetProp`/`GetProp`, `msvcrt.dll` al UCRT, DLL de 32 bits
  rechazadas en juegos de 64, salida de consola del juego en logcat.
- **Panorámica real para Cuphead** (Menú rápido → Imagen → Pantalla = **Panorámica**, por
  defecto): el juego dibuja más escenario a los lados, a resolución nativa, sin estirar ni
  recortar. HyperPlay parchea en memoria el límite 16:9 de la cámara del juego al cargarlo (los
  archivos del juego no se tocan). Probado en el teléfono: menús, mapa del mundo y un nivel de
  disparos llenan la pantalla 20:9 a 60 FPS. Detalle conocido: en algunos niveles una capa de
  bruma del cielo está dibujada para 16:9 y deja una costura suave cerca de los bordes antiguos.
- Menú rápido → Imagen → **Pantalla**: Original / Estirar / Ampliar, se guarda por juego. Llena
  la pantalla en juegos que dibujan en 16:9 con barras a los lados (Cuphead en un teléfono 20:9).
  Estirar ensancha la imagen; Ampliar mantiene la forma y recorta arriba y abajo. Probado en el
  teléfono con Cuphead.
- Cada juego corre ahora en su propio proceso. Forzar el cierre de un juego y volver a abrirlo
  (o abrir otro) ya no cierra la app. Quick Resume sigue funcionando (el juego suspendido se
  reanuda al instante); abrir otro juego cierra antes el suspendido.
- Launcher: ya no se pierde una pulsación de la cruceta cuando aparece la capa de RedMagic Game
  Space al abrir la app.
- Menú rápido: tocar ‹ ahora retrocede el ajuste y › lo avanza (antes cualquier toque avanzaba).
- Mando: las pulsaciones muy cortas se mantienen al menos 40 ms, así los juegos que leen el mando
  una vez por frame ya no las pierden.
- **Hollow Knight: Silksong arranca sin internet.** Alrededor de 1 de cada 5 arranques se quedaba
  en pantalla negra (con o sin internet; parecía un problema de red). Causa:
  `TryEnterCriticalSection` esperaba como `EnterCriticalSection` y el recolector de memoria de
  Mono, que la usa mientras pausa hilos, esperaba para siempre. Probado en el teléfono: 10 de 10
  arranques sin internet llegan al menú (~115 FPS) y una partida nueva se puede jugar.
- Arreglado un cierre cuando un juego abría un archivo con la pila de llamadas muy llena
  (Silksong al empezar partida): el código de panorámica de Cuphead usaba 64 KB de pila en cada
  apertura de archivo. El hilo del juego y los hilos que crean los juegos tienen ahora también
  más pila nativa.
- Un hilo que ejecuta código del juego desde dentro de una llamada nativa (callbacks,
  excepciones) vuelve a marcarse "en código nativo" al terminar, así pausarlo nunca espera para
  siempre.
- **Celeste tiene sonido, sin FMOD.** HyperPlay tiene ahora su propio FMOD Studio (código nativo,
  sin código de Firelight): lee los bancos de sonido del juego, decodifica sus sonidos Vorbis con
  Android y los mezcla a 48 kHz. Probado en el teléfono: ambiente del título (en bucle), música
  del menú y sonidos de la interfaz. Incluye: eventos y líneas de tiempo, variaciones aleatorias,
  eventos anidados, bucles y puntos de sostén, volumen y pausa de buses/VCA, música larga
  decodificada por bloques y un limitador suave. Probado en el prólogo: pasos, salto, aterrizaje,
  ambiente y música. Además: parámetros del juego (las capas de la música entran y salen como en
  PC; valores por defecto leídos del banco), sonidos elegidos por un parámetro y posición en
  estéreo (izquierda/derecha, más bajo fuera de la pantalla), los buses del mezclador y sus
  faders, snapshots (el menú de pausa baja la música y el ambiente; momentos de la historia bajan
  el ambiente) y música que cambia de sección cuando el juego lo indica. No hecho: efectos como la
  reverberación. Comprobado el guardado de Celeste: "Save and Quit" escribe la
  partida y se carga al reiniciar.
- Pantalla de carga: los juegos que dibujan con su propio Vulkan (Celeste) la quitan en cuanto
  muestran algo (antes podía durar hasta 45 s).
- Arreglado: Mono a veces abortaba (`mono-mmap-windows.c:132`) porque `VirtualQuery` daba por
  libre una región recién reservada; y el listado de una carpeta podía salir vacío al reutilizarse
  un número de handle (Celeste: "Missing Language Files").
- **Celeste vuelve a guardar.** Al cambiar una opción salía "Save Failed": Mono crea la carpeta
  `Saves` con una ruta larga de Win32 (`\\?\D:\Saves`) y HyperPlay no entendía el prefijo `\\?\`.
  Probado en el teléfono: una opción cambiada en Celeste sigue igual tras reiniciar el juego.
- Pantalla de carga: fuera el texto técnico del motor; ahora muestra una etapa amable
  ("Preparando el juego…", "Ya casi está…") y una frase divertida que cambia cada pocos segundos.
- Arreglada la detección de juegos .NET: los ejecutables mixtos C++/CLI (DmC: Devil May Cry) se
  arrancaban con Mono. Ahora solo usan Mono los ensamblados solo IL.
- Juegos de 32 bits: sus DLL se colocan siempre por debajo de 4 GB (antes algunas DLL de PhysX de
  DmC quedaban por encima) y la tabla de importaciones de una DLL de 32 bits se escribe con
  ranuras de 4 bytes (DmC se cerraba al cargar sus DLL de wxWidgets). DmC aún no arranca (faltan
  funciones de Windows).
- **Los informes de errores llegan al servidor.** Con tu permiso, los informes se envían al
  servidor de informes de HyperPlay, que abre un issue en GitHub (sin token en la app). Para
  ocupar poco en el teléfono: el registro de la app se vacía al enviar los informes, se guardan
  como mucho 20 informes sin enviar (de menos de 30 días) y solo los 5 últimos logs de sesión.

## 1.0.0

### English — first public version (alpha)

**Game status in this version**

| Game | Status |
|---|---|
| **Hollow Knight** | ✅ **Stable**: menus, saves, gameplay, benches and physics; 60-120 FPS |
| Hollow Knight: Silksong | 🧪 Starts; first tests only, not verified as playable yet |
| Cuphead | 🚧 Being optimized: menus, saves, world map and a level already run at ~120 FPS; the goal is to make it playable and stable for the next version |
| DmC: Devil May Cry | ❌ Not yet (needs 32-bit and Direct3D 9 support) |
| Celeste | ❌ Not yet (needs the .NET Framework runtime) |

Only **Hollow Knight** is considered stable in 1.0.0.

**Features**

- **Direct engine, no Wine or Linux**: Windows x86/x64 `.exe` games run directly on Android. PE
  loader, Win32 API interception and an x86/x64 → ARM64 JIT with an on-disk cache, real x87 FPU,
  SSE up to 4.2 and FMA.
- **GPU graphics**: Direct3D 11 → Vulkan (DXBC → SPIR-V shaders) on the Adreno 830, SGSR
  upscaling, CAS sharpening and HyperFG frame generation.
- **Handheld-console interface**: library with official covers, logos and game info from Steam,
  game page on a single screen, custom loading screen, Quick Resume (the game stays paused in
  memory) and an in-game quick menu (L3 + R3).
- **Controllers**: Xbox, PlayStation, GameSir, 8BitDo and generic HID over Bluetooth or USB, with
  automatic mapping, analog triggers and vibration (or the phone vibrates if the pad can't); full
  gamepad navigation in the launcher; touch controls appear when you touch the screen.
- **Performance and heat**: FPS cap (60/72/120/unlimited), render resolution, thermal protection
  that keeps the battery under 40 °C, and a performance overlay.
- **Your language**: the app follows the system language (English or Spanish), including the
  Steam game info.
- **Your folder**: no default folders; you choose where your games are.
- **In-app updates** from GitHub Releases, installed over the current version without losing
  settings, saves or caches.
- **Error log on the phone** and optional reports (asked on first launch). Automatic upload to
  GitHub is not enabled in this build yet.

**Requirements**: Android 12 or later, ARM64. Tuned for Snapdragon 8 Elite (tested on a
RedMagic 10S Pro). HyperPlay does not include games.

### Español — primera versión pública (alfa)

**Estado de los juegos en esta versión**

| Juego | Estado |
|---|---|
| **Hollow Knight** | ✅ **Estable**: menús, partidas, gameplay, bancos y física; 60-120 FPS |
| Hollow Knight: Silksong | 🧪 Arranca; solo primeras pruebas, aún sin verificar que sea jugable |
| Cuphead | 🚧 En optimización: menús, partidas, mapa del mundo y un nivel ya van a ~120 FPS; el objetivo es dejarlo jugable y estable para la próxima versión |
| DmC: Devil May Cry | ❌ Todavía no (necesita soporte de 32 bits y Direct3D 9) |
| Celeste | ❌ Todavía no (necesita el runtime de .NET Framework) |

En la 1.0.0 solo **Hollow Knight** se considera estable.

**Funciones**

- **Motor directo, sin Wine ni Linux**: los juegos `.exe` de Windows x86/x64 se ejecutan
  directamente en Android. Cargador PE, interceptación de la API Win32 y JIT x86/x64 → ARM64 con
  caché en disco, FPU x87 real, SSE hasta 4.2 y FMA.
- **Gráficos en la GPU**: Direct3D 11 → Vulkan (shaders DXBC → SPIR-V) en la Adreno 830,
  reescalado SGSR, nitidez CAS y generación de fotogramas HyperFG.
- **Interfaz de consola portátil**: biblioteca con portadas, logos e información oficial de Steam,
  ficha del juego en una sola pantalla, pantalla de carga propia, Quick Resume (el juego queda en
  pausa en memoria) y menú rápido dentro del juego (L3 + R3).
- **Mandos**: Xbox, PlayStation, GameSir, 8BitDo y HID genéricos por Bluetooth o USB, con mapeo
  automático, gatillos analógicos y vibración (o vibra el teléfono si el mando no puede);
  navegación completa con mando en el launcher; el mando táctil aparece al tocar la pantalla.
- **Rendimiento y temperatura**: tope de FPS (60/72/120/sin límite), resolución de render,
  protección térmica que mantiene la batería por debajo de 40 °C e indicador de rendimiento.
- **Tu idioma**: la app usa el idioma del sistema (inglés o español), también en la ficha de Steam.
- **Tu carpeta**: sin carpetas por defecto; tú eliges dónde están tus juegos.
- **Actualizaciones dentro de la app** desde GitHub Releases, instaladas por encima sin perder
  ajustes, partidas ni cachés.
- **Registro de errores en el teléfono** e informes opcionales (se pregunta en el primer arranque).
  El envío automático a GitHub aún no está activado en esta versión.

**Requisitos**: Android 12 o superior, ARM64. Optimizado para Snapdragon 8 Elite (probado en un
RedMagic 10S Pro). HyperPlay no incluye juegos.
