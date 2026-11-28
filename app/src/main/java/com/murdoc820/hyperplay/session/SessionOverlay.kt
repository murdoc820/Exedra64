package com.murdoc820.hyperplay.session

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.stringArrayResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import com.murdoc820.hyperplay.R
import kotlin.math.roundToInt
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.size
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.layout.ContentScale
import com.murdoc820.hyperplay.ui.ArtImage
import com.murdoc820.hyperplay.ui.Deck
import com.murdoc820.hyperplay.ui.Mac
import com.murdoc820.hyperplay.ui.MacGroup
import com.murdoc820.hyperplay.ui.MacSectionHeader
import com.murdoc820.hyperplay.ui.MacSeparator
import com.murdoc820.hyperplay.ui.MacStatusDot

enum class SessionPhase { LOADING, RUNNING, ENDED }

/**
 * Entradas del Quick Menu, en orden de navegación: primero las acciones (panel izquierdo) y luego
 * los ajustes (panel derecho, agrupados en Rendimiento / Imagen / Controles y pantalla).
 */
enum class QuickMenuItem {
    RESUME, QUICK_EXIT, FORCE_KILL,
    FPS_CAP, FRAME_GEN, THERMAL,
    UPSCALER, FSR_UPSCALE,
    OVERLAY, PHONE_RUMBLE;

    val isAction: Boolean get() = this == RESUME || this == QUICK_EXIT || this == FORCE_KILL
}

/** Estado observable de la sesión (solo se modifica desde el hilo principal). */
class SessionUiState {
    var phase by mutableStateOf(SessionPhase.LOADING)
    var title by mutableStateOf("")
    /** Arte del juego en caché (fondo y logo de la pantalla de carga). */
    var art by mutableStateOf<com.murdoc820.hyperplay.art.GameArt?>(null)
    var stage by mutableStateOf(LoadingStage.PE_LOAD)
    var elapsedMillis by mutableLongStateOf(0L)
    var logTail by mutableStateOf(emptyList<String>())
    var errorMessage by mutableStateOf<String?>(null)
    var exitCode by mutableIntStateOf(-1)
    var logPath by mutableStateOf("")

    var menuOpen by mutableStateOf(false)
    var menuIndex by mutableIntStateOf(0)
    var confirmKill by mutableStateOf(false)
    /** 0 = Cancelar, 1 = Forzar cierre. */
    var confirmIndex by mutableIntStateOf(0)
    var killing by mutableStateOf(false)

    var presentFps by mutableFloatStateOf(0f)
    /** FPS reales del juego (llamadas a Present), sin repeticiones ni sintéticos de HyperFG. */
    var gameFps by mutableFloatStateOf(0f)
    var receivedFps by mutableFloatStateOf(0f)
    var fsrUpscale by mutableStateOf(true)
    var notice by mutableStateOf<String?>(null)

    /** HUD de rendimiento (ver PerfOverlay / PerfMonitor). */
    var perfMode by mutableStateOf(PerfOverlayMode.DEFAULT)
    var perf by mutableStateOf(PerfSnapshot.EMPTY)

    /** Vibrar el teléfono si el mando no vibra. */
    var phoneRumble by mutableStateOf(false)

    /** Modo eficiente: tope elegido, protección térmica, escala (al iniciar) y decisión vigente. */
    var fpsCap by mutableStateOf(FpsCapOption.DEFAULT)
    var thermalProtection by mutableStateOf(ThermalGovernor.DEFAULT_ENABLED)
    var thermal by mutableStateOf(ThermalDecision(FpsCapOption.DEFAULT.fps, false, false, false))
    var upscaler by mutableStateOf(UpscalerMode.DEFAULT)
    var frameGen by mutableStateOf(FrameGenLogic.DEFAULT_ENABLED)
}

/**
 * Capa Compose encima del SurfaceView: pantalla de carga, Quick Menu y pantalla de fin de sesión.
 * Fuera de esos estados no dibuja nada ni consume toques.
 */
@Composable
fun SessionOverlay(
    state: SessionUiState,
    onMenuItem: (QuickMenuItem) -> Unit,
    onConfirmKill: (Boolean) -> Unit,
    onCloseMenu: () -> Unit,
    onExit: () -> Unit,
    /** Toque en ‹ (-1) o › (+1) de un ajuste. */
    onMenuStep: (QuickMenuItem, Int) -> Unit = { item, _ -> onMenuItem(item) },
) {
    Box(Modifier.fillMaxSize()) {
        when (state.phase) {
            SessionPhase.LOADING -> LoadingScreen(state)
            SessionPhase.ENDED -> EndedScreen(state, onExit)
            // HUD sin entrada de puntero: no bloquea el mando táctil ni el juego.
            SessionPhase.RUNNING -> PerfOverlay(
                state.perfMode, state.perf,
                hud = HudContext(
                    // El OSD muestra FPS reales del juego: se comparan con el tope real (sin x2 de HyperFG).
                    targetFps = FrameGenLogic.gameFps(state.thermal, state.frameGen),
                    thermal = state.thermal,
                    frameGen = state.frameGen,
                ),
            )
        }
        if (state.menuOpen && state.phase != SessionPhase.ENDED) {
            QuickMenu(state, onMenuItem, onConfirmKill, onCloseMenu, onMenuStep)
        }
    }
}

// ------------------------------------------------------------------ carga / fin

/**
 * Pantalla de carga de HyperPlay (no la pantalla negra del juego): arte panorámico del juego de
 * fondo, logo oficial, barra de progreso con el acento de la marca y la etapa actual. Se quita
 * cuando el juego empieza a dibujar de verdad (GameSessionLogic.contentStarted).
 */
@Composable
private fun LoadingScreen(state: SessionUiState) {
    val art = state.art
    Box(Modifier.fillMaxSize().background(Deck.BackdropBrush)) {
        val hero = art?.hero ?: art?.header
        if (hero != null) ArtImage(hero, Modifier.fillMaxSize())
        Box(Modifier.fillMaxSize().background(Brush.horizontalGradient(listOf(Color(0xF2060810), Color(0x99060810), Color(0x33060810)))))
        Box(Modifier.fillMaxSize().background(Brush.verticalGradient(listOf(Color.Transparent, Color(0xCC060810)))))
        // Marca arriba a la izquierda.
        Row(Modifier.align(Alignment.TopStart).padding(36.dp), verticalAlignment = Alignment.CenterVertically) {
            com.murdoc820.hyperplay.ui.HyperLogo(30.dp)
            Spacer(Modifier.width(10.dp))
            Text(androidx.compose.ui.res.stringResource(com.murdoc820.hyperplay.R.string.brand_upper), color = Deck.Text, fontSize = 14.sp, fontWeight = FontWeight.Black, letterSpacing = 2.sp)
        }
        Column(Modifier.align(Alignment.BottomStart).padding(start = 56.dp, bottom = 56.dp).widthIn(max = 620.dp)) {
            val logo = art?.logo
            val title = art?.info?.name?.takeIf { it.isNotBlank() } ?: state.title
            if (logo != null) {
                ArtImage(logo, Modifier.height(120.dp).widthIn(max = 460.dp).aspectRatio(2.6f),
                    contentScale = ContentScale.Fit, alignment = Alignment.CenterStart,
                    fallback = { LoadingTitle(title) })
            } else {
                LoadingTitle(title)
            }
            Spacer(Modifier.height(26.dp))
            LinearProgressIndicator(
                Modifier.width(420.dp).height(5.dp).clip(RoundedCornerShape(50)),
                color = Deck.Accent, trackColor = Color(0x332A3350),
            )
            Spacer(Modifier.height(12.dp))
            Text("${stringResource(state.stage.label)}  ·  ${GameSessionLogic.formatElapsed(state.elapsedMillis)}",
                color = Deck.Sub, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            // Frase amable que cambia cada pocos segundos (el log técnico queda en el archivo).
            val quips = stringArrayResource(R.array.loading_quips)
            if (quips.isNotEmpty()) {
                Spacer(Modifier.height(6.dp))
                Text(quips[GameSessionLogic.quipIndex(state.elapsedMillis, quips.size)], color = Deck.Dim,
                    fontSize = 13.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
        Text(stringResource(R.string.loading_quick_menu_hint), color = Deck.Dim, fontSize = 12.sp,
            modifier = Modifier.align(Alignment.BottomEnd).padding(36.dp))
    }
}

@Composable
private fun LoadingTitle(title: String) {
    Text(title, color = Deck.Text, fontSize = 40.sp, fontWeight = FontWeight.Black, maxLines = 2,
        overflow = TextOverflow.Ellipsis, lineHeight = 44.sp)
}

@Composable
private fun LogTail(lines: List<String>) {
    MacGroup {
        Column(Modifier.padding(horizontal = Mac.S3, vertical = Mac.S2)) {
            if (lines.isEmpty()) {
                Text("…", color = Mac.Tertiary, fontSize = Mac.Caption, fontFamily = FontFamily.Monospace)
            }
            for (line in lines) {
                Text(line, color = Mac.Secondary, fontSize = Mac.Caption, fontFamily = FontFamily.Monospace,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }
    }
}

@Composable
private fun EndedScreen(state: SessionUiState, onExit: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Mac.Window), contentAlignment = Alignment.Center) {
        Column(Modifier.widthIn(max = 600.dp).padding(Mac.S6)) {
            Text(state.title, color = Mac.Label, fontSize = Mac.Title, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(Mac.S2))
            val error = state.errorMessage
            Row(verticalAlignment = Alignment.CenterVertically) {
                MacStatusDot(if (error != null) Mac.Red else Mac.Secondary)
                Spacer(Modifier.width(Mac.S2))
                Text(
                    if (error != null) stringResource(R.string.ended_error)
                    else stringResource(R.string.ended_exit, GameSessionLogic.describeExitCode(state.exitCode)),
                    color = Mac.Label, fontSize = Mac.Headline,
                )
            }
            if (error != null) {
                Spacer(Modifier.height(Mac.S1))
                Text(error, color = Mac.Secondary, fontSize = Mac.Body)
            }
            Spacer(Modifier.height(Mac.S4))
            LogTail(state.logTail)
            if (state.logPath.isNotEmpty()) {
                Spacer(Modifier.height(Mac.S2))
                Text(state.logPath, color = Mac.Tertiary, fontSize = Mac.Caption, fontFamily = FontFamily.Monospace,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
            Spacer(Modifier.height(Mac.S5))
            ActionButton(stringResource(R.string.back_to_library), focused = true, style = ActionStyle.PRIMARY, onClick = onExit)
        }
    }
}

// ------------------------------------------------------------------ Quick Menu

@Composable
private fun QuickMenu(
    state: SessionUiState,
    onMenuItem: (QuickMenuItem) -> Unit,
    onConfirmKill: (Boolean) -> Unit,
    onCloseMenu: () -> Unit,
    onMenuStep: (QuickMenuItem, Int) -> Unit,
) {
    // Velo: un toque fuera del panel cierra el menú; también bloquea el mando táctil de debajo.
    Box(
        Modifier.fillMaxSize().background(Mac.Scrim)
            .pointerInput(Unit) { detectTapGestures { onCloseMenu() } }
            .padding(Mac.S4),
        contentAlignment = Alignment.Center,
    ) {
        Row(
            Modifier.widthIn(max = 900.dp).fillMaxHeight()
                .background(Mac.Sheet, Mac.RadiusLarge)
                .border(0.5.dp, Color(0x33FFFFFF), Mac.RadiusLarge)
                // Toques dentro del panel no cierran el menú.
                .pointerInput(Unit) { detectTapGestures { } },
        ) {
            ActionsPane(state, onMenuItem, onConfirmKill)
            Box(Modifier.width(0.5.dp).fillMaxHeight().background(Mac.Separator))
            SettingsPane(state, onMenuItem, onMenuStep, Modifier.weight(1f))
        }
    }
}

/** Panel izquierdo: juego, estado en vivo y acciones (o la confirmación de Forzar cierre). */
@Composable
private fun ActionsPane(
    state: SessionUiState,
    onMenuItem: (QuickMenuItem) -> Unit,
    onConfirmKill: (Boolean) -> Unit,
) {
    Column(
        Modifier.width(260.dp).fillMaxHeight()
            .background(Mac.Sidebar, RoundedCornerShape(topStart = 12.dp, bottomStart = 12.dp))
            .padding(Mac.S4),
    ) {
        Text(stringResource(R.string.info_quick_menu), color = Mac.Secondary, fontSize = Mac.Caption, fontWeight = FontWeight.SemiBold)
        Spacer(Modifier.height(2.dp))
        Text(state.title, color = Mac.Label, fontSize = Mac.Title, fontWeight = FontWeight.Bold,
            maxLines = 2, overflow = TextOverflow.Ellipsis)
        Spacer(Modifier.height(Mac.S3))
        LiveStatus(state)
        Spacer(Modifier.weight(1f))
        if (state.confirmKill) {
            Text(stringResource(R.string.kill_confirm_title), color = Mac.Label, fontSize = Mac.Headline, fontWeight = FontWeight.SemiBold)
            Spacer(Modifier.height(2.dp))
            Text(stringResource(R.string.kill_confirm_body), color = Mac.Secondary, fontSize = Mac.Callout)
            Spacer(Modifier.height(Mac.S3))
            ActionButton(stringResource(R.string.cancel), state.confirmIndex == 0, ActionStyle.SECONDARY) { onConfirmKill(false) }
            Spacer(Modifier.height(Mac.S2))
            ActionButton(stringResource(if (state.killing) R.string.closing else R.string.force_close), state.confirmIndex == 1,
                ActionStyle.DESTRUCTIVE) { onConfirmKill(true) }
        } else {
            ActionButton(stringResource(R.string.resume), state.menuIndex == QuickMenuItem.RESUME.ordinal, ActionStyle.PRIMARY) {
                onMenuItem(QuickMenuItem.RESUME)
            }
            Spacer(Modifier.height(Mac.S2))
            ActionButton(stringResource(R.string.quick_exit), state.menuIndex == QuickMenuItem.QUICK_EXIT.ordinal,
                ActionStyle.SECONDARY) { onMenuItem(QuickMenuItem.QUICK_EXIT) }
            Spacer(Modifier.height(Mac.S2))
            ActionButton(stringResource(R.string.force_close), state.menuIndex == QuickMenuItem.FORCE_KILL.ordinal,
                ActionStyle.DESTRUCTIVE) { onMenuItem(QuickMenuItem.FORCE_KILL) }
        }
        Spacer(Modifier.height(Mac.S3))
        Text(stringResource(R.string.menu_hint), color = Mac.Tertiary, fontSize = Mac.Caption)
    }
}

/** Estado en vivo con puntos de color y texto (no solo color). */
@Composable
private fun LiveStatus(state: SessionUiState) {
    // Tope real del juego (no el presentado) y FPS reales; si aún no hay muestra, cae al presentado.
    val target = FrameGenLogic.gameFps(state.thermal, state.frameGen)
    val fps = if (state.gameFps > 0f) state.gameFps else state.presentFps
    // Mismos umbrales de jugabilidad que el overlay de rendimiento (rojo solo por debajo de lo jugable).
    val fpsColor = when (PerfMonitorLogic.fpsLevel(fps.toFloat(), target)) {
        PerfLevel.BAD -> Mac.Red
        PerfLevel.WARN -> Mac.Orange
        else -> Mac.Green
    }
    StatusLine(fpsColor, if (target > 0) stringResource(R.string.fps_of, fps, target) else stringResource(R.string.fps_now, fps))
    if (state.frameGen && state.presentFps > 0f) {
        StatusLine(fpsColor, "HyperFG → ${state.presentFps.roundToInt()}")
    }
    val bat = state.perf.batteryTempC
    if (bat != null) {
        val c = when {
            bat < ThermalGovernor.THROTTLE_ON_C -> Mac.Green
            bat < ThermalGovernor.HOT_ON_C -> Mac.Orange
            else -> Mac.Red
        }
        StatusLine(c, stringResource(R.string.battery_temp, bat))
    }
    if (state.thermal.throttled) {
        StatusLine(
            Mac.Orange,
            if (state.frameGen) stringResource(R.string.thermal_fg, FrameGenLogic.THROTTLED_GAME_FPS * 2)
            else stringResource(R.string.thermal_cap, state.thermal.effectiveFps),
        )
    }
    state.notice?.let {
        Spacer(Modifier.height(Mac.S2))
        Text(it, color = Mac.Blue, fontSize = Mac.Callout)
    }
}

@Composable
private fun StatusLine(color: Color, text: String) {
    Row(Modifier.padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
        MacStatusDot(color)
        Spacer(Modifier.width(Mac.S2))
        Text(text, color = Mac.Label, fontSize = Mac.Callout)
    }
}

private enum class ActionStyle { PRIMARY, SECONDARY, DESTRUCTIVE }

@Composable
private fun ActionButton(text: String, focused: Boolean, style: ActionStyle, onClick: () -> Unit) {
    val shape = RoundedCornerShape(8.dp)
    val bg = when (style) {
        ActionStyle.PRIMARY -> Mac.Blue
        ActionStyle.SECONDARY -> Mac.Control
        ActionStyle.DESTRUCTIVE -> if (focused) Color(0x33FF453A) else Color.Transparent
    }
    Box(
        Modifier.fillMaxWidth()
            .then(if (focused) Modifier.border(3.dp, Mac.BlueSoft, shape).padding(1.dp) else Modifier)
            .then(if (focused && style != ActionStyle.PRIMARY) Modifier.border(2.dp, Mac.Blue, shape) else Modifier)
            .background(bg, shape)
            .clickable(remember { MutableInteractionSource() }, indication = null, onClick = onClick)
            .heightIn(min = 40.dp)
            .padding(horizontal = Mac.S3, vertical = 10.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text,
            color = if (style == ActionStyle.DESTRUCTIVE) Mac.Red else Mac.Label,
            fontSize = Mac.Body,
            fontWeight = if (style == ActionStyle.PRIMARY) FontWeight.SemiBold else FontWeight.Medium,
            maxLines = 1, overflow = TextOverflow.Ellipsis,
        )
    }
}

/** Panel derecho: ajustes agrupados (estilo Ajustes del Sistema), desplazable si no cabe. */
@Composable
private fun SettingsPane(
    state: SessionUiState,
    onMenuItem: (QuickMenuItem) -> Unit,
    onMenuStep: (QuickMenuItem, Int) -> Unit,
    modifier: Modifier,
) {
    Column(
        modifier.fillMaxHeight().verticalScroll(rememberScrollState()).padding(Mac.S4),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        val row: @Composable (QuickMenuItem) -> Unit = { item ->
            val s = settingFor(state, item)
            SettingRow(s, focused = state.menuIndex == item.ordinal && !state.confirmKill,
                onStep = { step -> onMenuStep(item, step) }) { onMenuItem(item) }
        }
        MacSectionHeader(stringResource(R.string.section_performance))
        MacGroup {
            row(QuickMenuItem.FPS_CAP); MacSeparator()
            row(QuickMenuItem.FRAME_GEN); MacSeparator()
            row(QuickMenuItem.THERMAL)
        }
        Spacer(Modifier.height(Mac.S3))
        MacSectionHeader(stringResource(R.string.section_image))
        MacGroup {
            row(QuickMenuItem.UPSCALER); MacSeparator()
            row(QuickMenuItem.FSR_UPSCALE)
        }
        Spacer(Modifier.height(Mac.S3))
        MacSectionHeader(stringResource(R.string.section_controls))
        MacGroup {
            row(QuickMenuItem.OVERLAY); MacSeparator()
            row(QuickMenuItem.PHONE_RUMBLE)
        }
    }
}

/** Contenido de una fila de ajuste. `locked` = no se puede cambiar ahora (en gris). */
private data class Setting(val label: String, val value: String, val hint: String? = null, val locked: Boolean = false)

@Composable
private fun settingFor(state: SessionUiState, item: QuickMenuItem): Setting = when (item) {
    QuickMenuItem.FPS_CAP -> if (!FrameGenLogic.fpsCapSelectable(state.frameGen)) {
        Setting(stringResource(R.string.set_fps_cap), stringResource(R.string.fps_cap_by_fg),
            hint = stringResource(R.string.fps_cap_by_fg_hint), locked = true)
    } else {
        val auto = state.thermal.throttled && state.thermal.effectiveFps != state.fpsCap.fps
        Setting(stringResource(R.string.set_fps_cap), stringResource(state.fpsCap.label),
            hint = if (auto) stringResource(R.string.fps_cap_heat, state.thermal.effectiveFps) else null)
    }
    QuickMenuItem.FRAME_GEN -> {
        val fps = FrameGenLogic.gameFps(state.thermal, true)
        Setting(
            stringResource(R.string.set_frame_gen),
            if (state.frameGen) stringResource(R.string.frame_gen_value, fps, fps * 2) else stringResource(R.string.state_off_f),
            hint = when {
                state.frameGen && state.thermal.throttled -> stringResource(R.string.frame_gen_throttled, fps * 2)
                state.frameGen -> stringResource(R.string.frame_gen_hint)
                else -> null
            },
        )
    }
    QuickMenuItem.THERMAL -> Setting(
        stringResource(R.string.set_thermal),
        stringResource(if (state.thermalProtection) R.string.state_on_f else R.string.state_off_f),
        hint = stringResource(R.string.thermal_hint),
    )
    QuickMenuItem.UPSCALER -> Setting(stringResource(R.string.set_upscaler), stringResource(state.upscaler.label))
    QuickMenuItem.FSR_UPSCALE -> Setting(stringResource(R.string.set_fsr), stringResource(if (state.fsrUpscale) R.string.state_on_f else R.string.state_off_f))
    QuickMenuItem.OVERLAY -> Setting(stringResource(R.string.set_overlay), stringResource(state.perfMode.label))
    QuickMenuItem.PHONE_RUMBLE -> Setting(
        stringResource(R.string.set_phone_rumble),
        stringResource(if (state.phoneRumble) R.string.yes else R.string.no),
        hint = stringResource(R.string.phone_rumble_hint),
    )
    else -> Setting(item.name, "")
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun SettingRow(s: Setting, focused: Boolean, onStep: (Int) -> Unit, onClick: () -> Unit) {
    val requester = remember { BringIntoViewRequester() }
    LaunchedEffect(focused) { if (focused) requester.bringIntoView() }
    val shape = RoundedCornerShape(8.dp)
    Row(
        Modifier.fillMaxWidth()
            .bringIntoViewRequester(requester)
            .background(if (focused) Mac.BlueSoft else Color.Transparent, shape)
            .then(if (focused) Modifier.border(2.dp, Mac.Blue, shape) else Modifier)
            .clickable(remember { MutableInteractionSource() }, indication = null, onClick = onClick)
            .padding(horizontal = Mac.S3, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(s.label, color = if (s.locked) Mac.Tertiary else Mac.Label, fontSize = Mac.Body)
            s.hint?.let { Text(it, color = Mac.Secondary, fontSize = Mac.Caption, maxLines = 1, overflow = TextOverflow.Ellipsis) }
        }
        Spacer(Modifier.width(Mac.S3))
        // Valor con flechas tipo selector de macOS (◀ ▶ del mando lo cambian). Al tocarlas, ‹ va
        // hacia atrás y › hacia delante; el resto de la fila avanza.
        @Composable
        fun Arrow(glyph: String, step: Int) = Text(
            glyph, color = Mac.Secondary, fontSize = 15.sp,
            modifier = Modifier
                .clickable(remember { MutableInteractionSource() }, indication = null) { onStep(step) }
                .padding(horizontal = 8.dp, vertical = 5.dp),
        )
        Row(
            Modifier.background(if (s.locked) Color.Transparent else Mac.Control, RoundedCornerShape(6.dp))
                .padding(horizontal = 2.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            if (!s.locked) Arrow("‹", -1)
            Text(
                s.value,
                color = if (s.locked) Mac.Tertiary else Mac.Label,
                fontSize = Mac.Callout, fontWeight = FontWeight.Medium,
                modifier = Modifier.padding(vertical = 5.dp),
                maxLines = 1,
            )
            if (!s.locked) Arrow("›", 1)
        }
    }
}
